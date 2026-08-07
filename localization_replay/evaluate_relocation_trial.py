#!/usr/bin/env python3
"""Same-sensor-timestamp Super-LIO evaluation against GLIM traj_imu.txt."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_data import (  # noqa: E402
    Trajectory,
    TrajectorySpec,
    load_odometry_only,
    load_trajectory,
    pose_matrix,
)


def trajectory_from_bag(path: Path, topic: str) -> Trajectory:
    series = load_odometry_only(path, (topic,))[topic]
    order = np.argsort(series.header_times, kind="stable")
    times = series.header_times[order]
    positions = series.positions[order]
    quaternions = series.quaternions_xyzw[order]
    times, unique = np.unique(times, return_index=True)
    positions = positions[unique]
    quaternions = quaternions[unique]
    if len(times) < 2:
        raise RuntimeError(f"{path}:{topic} contains fewer than two unique poses")
    return Trajectory(
        spec=TrajectorySpec(
            name=topic,
            path=path,
            bag_key=path.name,
            quat_order="xyzw",
            color=(0, 0, 0),
            time_mode="raw_sensor",
        ),
        raw_times=times.copy(),
        times=times,
        positions=positions,
        quaternions_xyzw=quaternions,
        alignment_note=f"{topic} header.stamp",
    )


def sample_many(trajectory: Trajectory, times: np.ndarray):
    positions = np.empty((len(times), 3), dtype=np.float64)
    quaternions = np.empty((len(times), 4), dtype=np.float64)
    for index, timestamp in enumerate(times):
        positions[index], quaternions[index], _ = trajectory.sample(float(timestamp))
    return positions, quaternions


def fixed_anchor(anchor: Trajectory, reference: Trajectory):
    start = max(float(anchor.times[0]), float(reference.times[0]))
    end = min(float(anchor.times[-1]), float(reference.times[-1]))
    if end <= start:
        raise RuntimeError("anchor bag and GLIM have no common sensor timestamp")
    anchor_position, anchor_quaternion, _ = anchor.sample(start)
    ref_position, ref_quaternion, _ = reference.sample(start)
    transform = (
        pose_matrix(anchor_position, anchor_quaternion)
        @ np.linalg.inv(pose_matrix(ref_position, ref_quaternion))
    )
    return transform, start


def summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def signed_summary(values: np.ndarray) -> dict[str, float | int | dict]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "absolute": summary(np.abs(values)),
    }


def window_metrics(
    estimate_positions: np.ndarray,
    estimate_quaternions: np.ndarray,
    reference_positions: np.ndarray,
    reference_quaternions: np.ndarray,
) -> dict:
    vectors = estimate_positions - reference_positions
    position = np.linalg.norm(vectors, axis=1)
    relative_rotation = (
        Rotation.from_quat(reference_quaternions).inv()
        * Rotation.from_quat(estimate_quaternions)
    )
    orientation = np.rad2deg(relative_rotation.magnitude())
    return {
        "position_error_m": summary(position),
        "orientation_error_deg": summary(orientation),
        "signed_axis_error_m": {
            axis: signed_summary(vectors[:, index])
            for index, axis in enumerate(("x", "y", "z"))
        },
    }


def relative_pose_error(
    times: np.ndarray,
    estimate_positions: np.ndarray,
    estimate_quaternions: np.ndarray,
    reference_positions: np.ndarray,
    reference_quaternions: np.ndarray,
    delta_sec: float,
) -> dict:
    translation_errors = []
    rotation_errors = []
    for left, timestamp in enumerate(times):
        right = int(np.searchsorted(times, timestamp + delta_sec, side="left"))
        if right >= len(times):
            break
        if abs(float(times[right] - timestamp) - delta_sec) > 0.1:
            continue
        estimate_left = pose_matrix(
            estimate_positions[left], estimate_quaternions[left]
        )
        estimate_right = pose_matrix(
            estimate_positions[right], estimate_quaternions[right]
        )
        reference_left = pose_matrix(
            reference_positions[left], reference_quaternions[left]
        )
        reference_right = pose_matrix(
            reference_positions[right], reference_quaternions[right]
        )
        estimate_delta = np.linalg.inv(estimate_left) @ estimate_right
        reference_delta = np.linalg.inv(reference_left) @ reference_right
        error = np.linalg.inv(reference_delta) @ estimate_delta
        translation_errors.append(float(np.linalg.norm(error[:3, 3])))
        rotation_errors.append(
            float(np.rad2deg(Rotation.from_matrix(error[:3, :3]).magnitude()))
        )
    return {
        "delta_sec": delta_sec,
        "translation_error_m": summary(np.asarray(translation_errors)),
        "orientation_error_deg": summary(np.asarray(rotation_errors)),
    }


def score(metrics: dict, coverage_ratio: float, minimum_coverage: float) -> dict:
    position = metrics["position_error_m"]
    z_abs = metrics["signed_axis_error_m"]["z"]["absolute"]
    orientation = metrics["orientation_error_deg"]
    # All main terms are metres. Z max deliberately has the largest individual
    # weight because vertical peaks are safety-critical for the current study.
    tracking_m = (
        0.15 * position["mean"]
        + 0.15 * position["rmse"]
        + 0.10 * position["p95"]
        + 0.10 * position["max"]
        + 0.10 * z_abs["mean"]
        + 0.10 * z_abs["p95"]
        + 0.10 * z_abs["p99"]
        + 0.20 * z_abs["max"]
        + 0.001 * orientation["p95"]
    )
    coverage_penalty_m = 100.0 * max(0.0, minimum_coverage - coverage_ratio)
    return {
        "value": float(tracking_m + coverage_penalty_m),
        "unit": "m-equivalent (lower is better)",
        "tracking_component_m": float(tracking_m),
        "coverage_penalty_m": float(coverage_penalty_m),
        "minimum_coverage_ratio": float(minimum_coverage),
        "weights": {
            "position_mean": 0.15,
            "position_rmse": 0.15,
            "position_p95": 0.10,
            "position_max": 0.10,
            "abs_z_mean": 0.10,
            "abs_z_p95": 0.10,
            "abs_z_p99": 0.10,
            "abs_z_max": 0.20,
            "orientation_p95_m_per_deg": 0.001,
        },
    }


def evaluate(args: argparse.Namespace) -> dict:
    estimate = trajectory_from_bag(args.estimated_bag, args.estimated_topic)
    anchor = trajectory_from_bag(args.anchor_bag, args.anchor_topic)
    reference = load_trajectory(
        TrajectorySpec(
            name="GLIM corrected IMU",
            path=args.glim_trajectory,
            bag_key="glim",
            quat_order="xyzw",
            color=(0, 0, 0),
            time_mode="raw_sensor",
        )
    )
    transform, anchor_timestamp = fixed_anchor(anchor, reference)
    rotation = Rotation.from_matrix(transform[:3, :3])

    common_start = max(float(estimate.times[0]), float(reference.times[0]))
    common_end = min(float(estimate.times[-1]), float(reference.times[-1]))
    expected_start = max(float(anchor.times[0]), float(reference.times[0]))
    expected_end = min(float(anchor.times[-1]), float(reference.times[-1]))
    if common_end <= common_start:
        raise RuntimeError("estimated trajectory and GLIM have no common interval")

    mask = (reference.times >= common_start) & (reference.times <= common_end)
    times = reference.times[mask]
    # Reject interpolation across a missing-output gap.
    right = np.searchsorted(estimate.times, times, side="left")
    valid = (right > 0) & (right < len(estimate.times))
    gaps = np.full(len(times), np.inf, dtype=np.float64)
    valid_indices = np.flatnonzero(valid)
    gaps[valid_indices] = (
        estimate.times[right[valid_indices]]
        - estimate.times[right[valid_indices] - 1]
    )
    valid &= gaps <= args.max_interpolation_gap_sec
    times = times[valid]
    if len(times) < 2:
        raise RuntimeError("fewer than two valid same-timestamp pose matches")

    estimate_positions, estimate_quaternions = sample_many(estimate, times)
    reference_positions, reference_quaternions = sample_many(reference, times)
    reference_positions = rotation.apply(reference_positions) + transform[:3, 3]
    reference_quaternions = (
        rotation * Rotation.from_quat(reference_quaternions)
    ).as_quat()

    elapsed = times - common_start
    steady_mask = elapsed >= args.score_trim_sec
    if np.count_nonzero(steady_mask) < 2:
        raise RuntimeError("score trim leaves fewer than two matched poses")
    full_metrics = window_metrics(
        estimate_positions,
        estimate_quaternions,
        reference_positions,
        reference_quaternions,
    )
    scoring_metrics = window_metrics(
        estimate_positions[steady_mask],
        estimate_quaternions[steady_mask],
        reference_positions[steady_mask],
        reference_quaternions[steady_mask],
    )
    expected_duration = max(expected_end - expected_start, 1.0e-9)
    coverage_ratio = float((common_end - common_start) / expected_duration)
    objective = score(scoring_metrics, coverage_ratio, args.minimum_coverage)
    rpe = relative_pose_error(
        times[steady_mask],
        estimate_positions[steady_mask],
        estimate_quaternions[steady_mask],
        reference_positions[steady_mask],
        reference_quaternions[steady_mask],
        args.rpe_delta_sec,
    )
    return {
        "method": {
            "time_alignment": (
                "raw sensor/header timestamps; GLIM timestamps sampled with "
                "Super-LIO interpolation; no fitted time offset"
            ),
            "spatial_alignment": (
                "one candidate-independent fixed T_map_glim from anchor bag "
                "and GLIM at their first common sensor timestamp"
            ),
            "estimated_pose_frame": "Livox IMU (/lio/odom)",
            "reference_pose_frame": "GLIM traj_imu.txt IMU pose",
            "score_trim_sec": float(args.score_trim_sec),
        },
        "sources": {
            "estimated_bag": str(args.estimated_bag.resolve()),
            "estimated_topic": args.estimated_topic,
            "anchor_bag": str(args.anchor_bag.resolve()),
            "anchor_topic": args.anchor_topic,
            "glim_trajectory": str(args.glim_trajectory.resolve()),
        },
        "fixed_transform_map_from_glim": transform.tolist(),
        "anchor_sensor_timestamp": float(anchor_timestamp),
        "data": {
            "estimate_pose_count": int(len(estimate.times)),
            "glim_pose_count": int(len(reference.times)),
            "matched_pose_count": int(len(times)),
            "scored_pose_count": int(np.count_nonzero(steady_mask)),
            "common_start_sensor_time": common_start,
            "common_end_sensor_time": common_end,
            "common_duration_sec": float(common_end - common_start),
            "expected_duration_sec": float(expected_duration),
            "coverage_ratio": coverage_ratio,
        },
        "full_interval": full_metrics,
        "scoring_interval": scoring_metrics,
        "relative_pose_error": rpe,
        "objective": objective,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimated-bag", type=Path, required=True)
    parser.add_argument("--anchor-bag", type=Path, required=True)
    parser.add_argument("--glim-trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--estimated-topic", default="/lio/odom")
    parser.add_argument("--anchor-topic", default="/lio/odom")
    parser.add_argument("--score-trim-sec", type=float, default=0.0)
    parser.add_argument("--rpe-delta-sec", type=float, default=1.0)
    parser.add_argument("--max-interpolation-gap-sec", type=float, default=0.1)
    parser.add_argument("--minimum-coverage", type=float, default=0.90)
    args = parser.parse_args()
    if args.score_trim_sec < 0.0:
        parser.error("--score-trim-sec must be non-negative")
    if not 0.0 < args.minimum_coverage <= 1.0:
        parser.error("--minimum-coverage must be in (0, 1]")
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    compact = {
        "output": str(args.output),
        "score": result["objective"]["value"],
        "coverage": result["data"]["coverage_ratio"],
        "position_mean_m": result["scoring_interval"]["position_error_m"]["mean"],
        "position_max_m": result["scoring_interval"]["position_error_m"]["max"],
        "z_abs_max_m": result["scoring_interval"]["signed_axis_error_m"]["z"]["absolute"]["max"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()

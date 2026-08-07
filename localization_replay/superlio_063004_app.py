#!/usr/bin/env python3
"""Viser replay for dated G1 localization runs."""

from __future__ import annotations

import argparse
import asyncio
from functools import partial
import os
from pathlib import Path
import threading
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

from app import (
    LiveAugmentedMapReceiver,
    _joint_figure,
    _pose_figure,
    _set_figure_cursor,
    _update_joint_figure,
    _update_pose_figure,
)
from replay_data import (
    apply_spatial_transform,
    BagData,
    G1_JOINT_NAMES,
    matrix_pose,
    pose_matrix,
    quaternion_xyzw_to_wxyz,
    Trajectory,
    TrajectorySpec,
    load_bag,
    load_odometry_only,
    load_pcd,
    load_trajectory,
    remove_trajectory_pose_matches,
)


PROJECT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_DIR / "assets"
DEFAULT_ROSBAG_ROOT = Path(
    os.environ.get("LOCALIZATION_REPLAY_ROSBAG_ROOT", "/workspace/rosbag")
).expanduser()
DEFAULT_MAP = ASSETS_DIR / "dump7_xyzi_clean.pcd"
DEFAULT_URDF = ASSETS_DIR / "g1_description/g1_29dof_rev_1_0.urdf"
ANKLE_ROLL_SOLE_Z = -0.03540915


def _format_run_time(run_id: str) -> str:
    return (
        f"{run_id[0:2]}:{run_id[2:4]}:{run_id[4:6]}"
        if len(run_id) == 6 and run_id.isdigit()
        else run_id
    )


def _experiment_directory_name(prefix: str, run_date: str, run_id: str) -> str:
    if len(run_id) != 6 or not run_id.isdigit():
        raise ValueError(f"run-id must be HHMMSS, got {run_id!r}")
    date_parts = run_date.split("_")
    if len(date_parts) != 3 or any(
        not part.isdigit() for part in date_parts
    ):
        raise ValueError(
            f"run-date must be YYYY_MM_DD, got {run_date!r}"
        )
    return f"{prefix}_{run_date}-{run_id[0:2]}_{run_id[2:4]}_{run_id[4:6]}"


def _glim_directory_name(run_id: str, run_date: str = "2026_08_04") -> str:
    return _experiment_directory_name("glim_rosbag2", run_date, run_id)


def _source_directory_name(run_id: str, run_date: str = "2026_08_04") -> str:
    return _experiment_directory_name("rosbag2", run_date, run_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay Super-LIO, or compare GLIM corrected IMU and Super-LIO IMU "
            "poses from one G1 run with synchronized timestamps."
        )
    )
    parser.add_argument("--run-id", default="063004", help="Run time as HHMMSS")
    parser.add_argument(
        "--run-date",
        default="2026_08_04",
        help="Experiment date as YYYY_MM_DD (default: 2026_08_04)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_ROSBAG_ROOT,
        help=(
            "Directory containing rosbag2_* and glim_rosbag2_* directories. "
            "Default: "
            "$LOCALIZATION_REPLAY_ROSBAG_ROOT or /workspace/rosbag."
        ),
    )
    parser.add_argument("--bag", type=Path, default=None)
    parser.add_argument(
        "--source-bag",
        type=Path,
        default=None,
        help=(
            "Original experiment bag. It supplies the recorded live /lio/odom "
            "and the fixed one-pose map-frame anchor."
        ),
    )
    parser.add_argument(
        "--glim-trajectory",
        type=Path,
        default=None,
        help=(
            "GLIM traj_imu.txt fallback. It is used only when the bag has no "
            "/glim_ros/odom_corrected topic."
        ),
    )
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--max-map-points", type=int, default=1_000_000)
    parser.add_argument("--max-aug-map-points", type=int, default=200_000)
    parser.add_argument("--max-cloud-world-points", type=int, default=200_000)
    parser.add_argument("--map-point-size", type=float, default=0.01)
    parser.add_argument("--aug-map-point-size", type=float, default=0.01)
    parser.add_argument("--live-aug-map-topic", default="/augmented_map")
    parser.add_argument("--no-static-map", action="store_true")
    parser.add_argument("--no-live-aug-map", action="store_true")
    parser.add_argument("--no-spatial-align", action="store_true")
    parser.add_argument(
        "--post-recovery-align-steps",
        type=int,
        default=0,
        help=(
            "After the startup error recovers, average this many fixed "
            "T_MAP_GLIM samples and freeze the result. Zero disables this "
            "relative-tracking alignment."
        ),
    )
    parser.add_argument(
        "--recovery-position-threshold",
        type=float,
        default=0.08,
        help="Position-error threshold [m] used to detect startup recovery.",
    )
    parser.add_argument(
        "--recovery-orientation-threshold",
        type=float,
        default=3.0,
        help="Orientation-error threshold [deg] used to detect startup recovery.",
    )
    parser.add_argument(
        "--recovery-consecutive-steps",
        type=int,
        default=30,
        help="Consecutive stable GLIM samples required before N-step alignment.",
    )
    parser.add_argument(
        "--superlio-only",
        action="store_true",
        help=(
            "Replay only the selected bag's /lio/odom trajectory. GLIM dump, "
            "GLIM alignment, and estimator error plots are not required."
        ),
    )
    parser.add_argument("--no-robot", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _glim_directory_name(args.run_id, args.run_date)
    if args.post_recovery_align_steps < 0:
        parser.error("--post-recovery-align-steps must be non-negative")
    if args.recovery_consecutive_steps <= 0:
        parser.error("--recovery-consecutive-steps must be positive")
    if (
        args.recovery_position_threshold <= 0.0
        or args.recovery_orientation_threshold <= 0.0
    ):
        parser.error("recovery thresholds must be positive")
    if args.post_recovery_align_steps and (
        args.no_spatial_align or args.superlio_only
    ):
        parser.error(
            "post-recovery alignment requires GLIM and the initial spatial anchor"
        )
    if args.source_bag is None:
        args.source_bag = args.data_root.expanduser() / _source_directory_name(
            args.run_id, args.run_date
        )
    if args.bag is None:
        args.bag = args.source_bag
    return args


def _resolve_glim_path(
    requested: Path | None,
    run_id: str,
    run_date: str,
    bag_path: Path,
    data_root: Path,
) -> Path:
    if requested is not None:
        if not requested.is_file():
            raise FileNotFoundError(f"GLIM trajectory not found: {requested}")
        return requested
    directory_name = _glim_directory_name(run_id, run_date)
    glim_directory = data_root.expanduser() / directory_name
    candidates = (
        bag_path / "traj_imu.txt",
        glim_directory / "traj_imu.txt",
        glim_directory / "tmp" / directory_name / "traj_imu.txt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        "The bag has no /glim_ros/odom_corrected and no GLIM traj_imu fallback "
        f"was found. Searched:\n  {searched}"
    )


def _trajectory_from_odometry(
    name: str,
    bag_key: str,
    topic: str,
    series,
    color: tuple[int, int, int],
) -> Trajectory:
    order = np.argsort(series.header_times, kind="stable")
    times = np.asarray(series.header_times[order], dtype=np.float64)
    positions = np.asarray(series.positions[order], dtype=np.float64)
    quaternions = np.asarray(series.quaternions_xyzw[order], dtype=np.float64)
    quaternions /= np.maximum(
        np.linalg.norm(quaternions, axis=1, keepdims=True), 1.0e-12
    )
    spec = TrajectorySpec(
        name=name,
        path=Path(f"bag:{topic}"),
        bag_key=bag_key,
        quat_order="xyzw",
        color=color,
        match_topic=topic,
        time_mode="raw_sensor",
    )
    return Trajectory(
        spec=spec,
        raw_times=times.copy(),
        times=times,
        positions=positions,
        quaternions_xyzw=quaternions,
        alignment_note=f"{topic} header.stamp (original sensor clock)",
    )


def _choose_lio_topic(bag: BagData) -> str:
    for topic in ("/lio/imu/odom", "/lio/odom"):
        if topic in bag.odometry:
            return topic
    raise KeyError("Neither /lio/imu/odom nor /lio/odom exists in the bag")


def _required_tf(
    bag: BagData, parent: str, child: str, timestamp: float
) -> np.ndarray:
    """Return ROS T_parent_child, failing instead of hiding a broken TF chain."""
    transform = bag.transform(parent, child, timestamp)
    if transform is None:
        raise KeyError(f"Required TF {parent} -> {child} is missing from the bag")
    return transform


def _audit_sensor_tf(bag: BagData) -> None:
    """Check the bag's ambiguous imu/lidar alias against the physical Livox TF."""
    timestamp = bag.start_time
    frame_from_imu = _required_tf(
        bag, "mid360_link_frame", "mid360_link_IMU", timestamp
    )
    physical_imu_from_lidar = np.linalg.inv(frame_from_imu)
    alias_imu_from_lidar = bag.transform("imu", "lidar", timestamp)
    if alias_imu_from_lidar is None:
        print(
            "  Sensor TF: using inverse(mid360_link_frame -> mid360_link_IMU); "
            "the imu -> lidar alias is absent.",
            flush=True,
        )
        return

    translation_error = float(
        np.linalg.norm(
            alias_imu_from_lidar[:3, 3] - physical_imu_from_lidar[:3, 3]
        )
    )
    rotation_error = float(
        np.rad2deg(
            Rotation.from_matrix(
                alias_imu_from_lidar[:3, :3].T
                @ physical_imu_from_lidar[:3, :3]
            ).magnitude()
        )
    )
    if translation_error > 1.0e-6 or rotation_error > 1.0e-4:
        print(
            "  WARNING: recorded imu -> lidar disagrees with the physical "
            "mid360_link_frame -> mid360_link_IMU chain "
            f"(translation {translation_error:.4f} m, rotation "
            f"{rotation_error:.4f} deg); ignoring the alias TF.",
            flush=True,
        )
    else:
        print("  Sensor TF aliases are consistent.", flush=True)


def _load_static_map(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    map_points = np.empty((0, 3), dtype=np.float32)
    map_colors = np.empty((0, 3), dtype=np.uint8)
    if not args.no_static_map:
        print("[4/4] Loading static PCD map...", flush=True)
        map_points, map_colors, _ = load_pcd(
            args.map, max_points=args.max_map_points
        )
    else:
        print(
            "[4/4] Static PCD disabled; bag /augmented_map remains available.",
            flush=True,
        )
    return map_points, map_colors


def load_inputs(args: argparse.Namespace):
    print(
        f"[1/4] Loading experiment {args.run_id} lowstate, TF, odometry, and map index...",
        flush=True,
    )
    if not args.source_bag.exists():
        raise FileNotFoundError(
            f"Original replay source bag not found: {args.source_bag}"
        )
    same_as_original = args.bag.resolve() == args.source_bag.resolve()
    # For a clean offline evaluation bag, only /lio/odom is newly generated.
    # LowState, physical TF and augmented_map remain the immutable source-run
    # context, avoiding duplicated multi-gigabyte sensor data and mixed topics.
    context_bag_path = args.bag if same_as_original else args.source_bag
    bag = load_bag(
        args.run_id,
        context_bag_path,
        odometry_topics=(
            "/glim_ros/odom_corrected",
            "/glim_ros/odom",
            "/lio/imu/odom",
            "/lio/odom",
            "/odometry/odom",
        ),
        pointcloud_topics=("/augmented_map", "/lio/cloud_world"),
    )
    _required_tf(bag, "map", "world", bag.start_time)
    _required_tf(bag, "mid360_link", "mid360_link_frame", bag.start_time)
    _audit_sensor_tf(bag)
    if same_as_original:
        evaluation_odometry = bag.odometry
    else:
        evaluation_odometry = load_odometry_only(
            args.bag,
            topics=("/lio/imu/odom", "/lio/odom"),
        )
    lio_topic = next(
        (
            topic
            for topic in ("/lio/imu/odom", "/lio/odom")
            if topic in evaluation_odometry
        ),
        None,
    )
    if lio_topic is None:
        raise KeyError(
            f"Neither /lio/imu/odom nor /lio/odom exists in {args.bag}"
        )
    lio = _trajectory_from_odometry(
        "Super LIO IMU",
        bag.key,
        lio_topic,
        evaluation_odometry[lio_topic],
        (67, 190, 113),
    )
    source_odometry = bag.odometry
    if same_as_original:
        lio.filter_note = (
            "original experiment /lio/odom; no re-recorded stream and no filtering"
        )
        print(
            f"  Super-LIO source: original live stream with {len(lio.times):,} "
            f"poses from {args.bag}.",
            flush=True,
        )
    else:
        print(
            f"  Super-LIO source: clean/offline evaluation stream with "
            f"{len(lio.times):,} poses from {args.bag}; replay context comes "
            f"from {args.source_bag}.",
            flush=True,
        )

    if not same_as_original and lio_topic == "/lio/odom":
        if lio_topic not in source_odometry:
            raise KeyError(f"{lio_topic} is absent from {args.source_bag}")
        removed = remove_trajectory_pose_matches(
            lio, source_odometry[lio_topic]
        )
        if removed == 0:
            lio.filter_note = (
                "clean offline localization /lio/odom; no exact original "
                "replay poses were mixed into this topic"
            )
            print(
                "  Super-LIO stream separation: clean evaluation topic; "
                "no replayed original /lio/odom poses needed removal.",
                flush=True,
            )
        else:
            print(
                f"  Super-LIO stream separation: removed {removed:,} poses "
                f"replayed from {args.source_bag}; kept {len(lio.times):,} "
                "localization-only poses.",
                flush=True,
            )

    if args.superlio_only:
        print(
            "[2/4] Super-LIO-only mode; skipping GLIM source and alignment.",
            flush=True,
        )
        print("[3/4] Fixed GLIM frame anchor not required.", flush=True)
        spatial_alignment = np.eye(4, dtype=np.float64)
        map_points, map_colors = _load_static_map(args)
        print(
            f"  source: {lio_topic} in {args.bag}\n"
            f"  replay: {lio.name} only\n"
            f"  sensor interval: {lio.times[0]:.9f}..{lio.times[-1]:.9f} "
            f"({lio.times[-1] - lio.times[0]:.3f} s)\n"
            f"  static map points: {len(map_points):,}",
            flush=True,
        )
        return (
            bag,
            [lio],
            lio_topic,
            "disabled (--superlio-only)",
            spatial_alignment,
            map_points,
            map_colors,
        )

    print("[2/4] Selecting the like-for-like GLIM IMU pose source...", flush=True)
    glim_path = _resolve_glim_path(
        args.glim_trajectory,
        args.run_id,
        args.run_date,
        args.bag,
        args.data_root,
    )
    glim_lidar_path = glim_path.with_name("traj_lidar.txt")
    if not glim_lidar_path.is_file():
        raise FileNotFoundError(
            "Fixed map-frame anchoring requires the matching "
            f"traj_lidar.txt: {glim_lidar_path}"
        )
    glim_lidar = load_trajectory(
        TrajectorySpec(
            name="GLIM LiDAR calibration pose",
            path=glim_lidar_path,
            bag_key=bag.key,
            quat_order="xyzw",
            color=(239, 83, 80),
            time_mode="raw_sensor",
        )
    )

    if "/glim_ros/odom_corrected" in bag.odometry:
        glim_source = "/glim_ros/odom_corrected in the selected bag"
        glim = _trajectory_from_odometry(
            "GLIM corrected IMU",
            bag.key,
            "/glim_ros/odom_corrected",
            bag.odometry["/glim_ros/odom_corrected"],
            (239, 83, 80),
        )
    else:
        glim_source = f"GLIM dump traj_imu.txt fallback: {glim_path}"
        glim = load_trajectory(
            TrajectorySpec(
                name="GLIM corrected IMU (dump)",
                path=glim_path,
                bag_key=bag.key,
                quat_order="xyzw",
                color=(239, 83, 80),
                time_mode="raw_sensor",
            )
        )
        glim.alignment_note = "traj_imu timestamp (original sensor clock)"
        print(
            "  WARNING: /glim_ros/odom_corrected is absent from this bag; "
            f"using the matching {_format_run_time(args.run_id)} GLIM dump: "
            f"{glim_path}",
            flush=True,
        )

    trajectories = [glim, lio]
    common_start = max(item.times[0] for item in trajectories)
    common_end = min(item.times[-1] for item in trajectories)
    if common_end <= common_start:
        raise RuntimeError("GLIM and Super LIO have no common sensor timestamp interval")

    if "/odometry/odom" not in source_odometry:
        raise KeyError(
            "Fixed map-frame anchoring requires /odometry/odom in "
            f"the original source bag: {args.source_bag}"
        )
    map_lidar = _trajectory_from_odometry(
        "Map-frame GLIM LiDAR calibration pose",
        bag.key,
        "/odometry/odom",
        source_odometry["/odometry/odom"],
        (239, 83, 80),
    )

    print("[3/4] Computing fixed one-pose map <- GLIM frame anchor...", flush=True)
    spatial_alignment = np.eye(4, dtype=np.float64)
    if not args.no_spatial_align:
        calibration_start = max(map_lidar.times[0], glim_lidar.times[0])
        calibration_end = min(map_lidar.times[-1], glim_lidar.times[-1])
        if calibration_end <= calibration_start:
            raise RuntimeError(
                "/odometry/odom and traj_lidar.txt have no common sensor "
                "timestamp for map-frame calibration"
            )
        map_position, map_quaternion, _ = map_lidar.sample(calibration_start)
        glim_lidar_position, glim_lidar_quaternion, _ = glim_lidar.sample(
            calibration_start
        )
        if same_as_original:
            lio_position, lio_quaternion, _ = lio.sample(calibration_start)
            map_from_world = _required_tf(
                bag, "map", "world", bag.start_time
            )
            observed_imu_from_lidar = np.linalg.inv(
                map_from_world @ pose_matrix(lio_position, lio_quaternion)
            ) @ pose_matrix(map_position, map_quaternion)
            expected_imu_from_lidar = np.linalg.inv(
                _required_tf(
                    bag,
                    "mid360_link_frame",
                    "mid360_link_IMU",
                    bag.start_time,
                )
            )
            source_translation_error = float(
                np.linalg.norm(
                    observed_imu_from_lidar[:3, 3]
                    - expected_imu_from_lidar[:3, 3]
                )
            )
            source_rotation_error = float(
                np.rad2deg(
                    Rotation.from_matrix(
                        observed_imu_from_lidar[:3, :3].T
                        @ expected_imu_from_lidar[:3, :3]
                    ).magnitude()
                )
            )
            if (
                source_translation_error > 1.0e-5
                or source_rotation_error > 1.0e-4
            ):
                raise RuntimeError(
                    "Original /odometry/odom is not the expected "
                    "/lio/odom @ T_IMU_LIDAR stream: translation error "
                    f"{source_translation_error:.6f} m, rotation error "
                    f"{source_rotation_error:.6f} deg"
                )
            print(
                "  Source frame audit: /odometry/odom equals recorded live "
                "/lio/odom @ T_IMU_LIDAR at the anchor pose.",
                flush=True,
            )
        spatial_alignment = pose_matrix(
            map_position, map_quaternion
        ) @ np.linalg.inv(
            pose_matrix(glim_lidar_position, glim_lidar_quaternion)
        )
        note = (
            "fixed one-pose T_MAP_GLIM = "
            "T_MAP_LIDAR(/odometry/odom) @ "
            "inverse(T_GLIM_LIDAR(traj_lidar.txt)) at sensor timestamp "
            f"{calibration_start:.9f}; /odometry/odom is the recorded live "
            "Super-LIO pose composed with T_IMU_LIDAR; no trajectory fitting "
            "or per-frame alignment was used"
        )
        apply_spatial_transform(glim, spatial_alignment, note)
        if args.post_recovery_align_steps > 0:
            correction, recovery_elapsed, alignment_end_elapsed = (
                apply_post_recovery_alignment(
                    glim,
                    lio,
                    args.post_recovery_align_steps,
                    args.recovery_position_threshold,
                    args.recovery_orientation_threshold,
                    args.recovery_consecutive_steps,
                )
            )
            spatial_alignment = correction @ spatial_alignment
            correction_translation = float(
                np.linalg.norm(correction[:3, 3])
            )
            correction_rotation = float(
                np.rad2deg(
                    Rotation.from_matrix(correction[:3, :3]).magnitude()
                )
            )
            print(
                "  Post-recovery fixed alignment: "
                f"recovered at +{recovery_elapsed:.3f} s, "
                f"N={args.post_recovery_align_steps} through "
                f"+{alignment_end_elapsed:.3f} s; correction "
                f"{correction_translation:.4f} m / "
                f"{correction_rotation:.3f} deg.",
                flush=True,
            )
            glim.post_recovery_alignment_end_elapsed = alignment_end_elapsed
    else:
        glim.spatial_alignment_note = "disabled by --no-spatial-align"

    map_points, map_colors = _load_static_map(args)

    print(
        f"  source: {glim_source}\n"
        f"  compare: {glim.name} <-> {lio.name} ({lio_topic})\n"
        f"  common sensor interval: {common_start:.9f}..{common_end:.9f} "
        f"({common_end - common_start:.3f} s)\n"
        f"  static map points: {len(map_points):,}",
        flush=True,
    )
    return (
        bag,
        trajectories,
        lio_topic,
        glim_source,
        spatial_alignment,
        map_points,
        map_colors,
    )


def _sample_many(
    trajectory: Trajectory, sensor_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positions = []
    quaternions = []
    for timestamp in sensor_times:
        position, quaternion, _ = trajectory.sample(float(timestamp))
        positions.append(position)
        quaternions.append(quaternion)
    return np.stack(positions), np.stack(quaternions)


def apply_post_recovery_alignment(
    glim: Trajectory,
    lio: Trajectory,
    steps: int,
    position_threshold: float,
    orientation_threshold: float,
    consecutive_steps: int,
) -> tuple[np.ndarray, float, float]:
    """Estimate one fixed SE(3) correction after the startup spike settles."""
    common_start = max(float(glim.times[0]), float(lio.times[0]))
    common_end = min(float(glim.times[-1]), float(lio.times[-1]))
    mask = (glim.times >= common_start) & (glim.times <= common_end)
    sensor_times = np.unique(glim.times[mask])
    if len(sensor_times) < consecutive_steps + steps:
        raise RuntimeError(
            "Not enough common samples for post-recovery alignment: "
            f"need at least {consecutive_steps + steps}, got {len(sensor_times)}"
        )

    glim_position, glim_quaternion = _sample_many(glim, sensor_times)
    lio_position, lio_quaternion = _sample_many(lio, sensor_times)
    position_error = np.linalg.norm(lio_position - glim_position, axis=1)
    orientation_error = np.rad2deg(
        (
            Rotation.from_quat(lio_quaternion).inv()
            * Rotation.from_quat(glim_quaternion)
        ).magnitude()
    )
    stable = (
        (position_error <= position_threshold)
        & (orientation_error <= orientation_threshold)
    )
    recovery_index = None
    for index in range(len(stable) - consecutive_steps + 1):
        if bool(np.all(stable[index : index + consecutive_steps])):
            recovery_index = index
            break
    if recovery_index is None:
        raise RuntimeError(
            "Startup recovery was not found: no "
            f"{consecutive_steps}-sample interval stayed below "
            f"{position_threshold:.3f} m and {orientation_threshold:.3f} deg"
        )

    alignment_end = recovery_index + steps
    if alignment_end > len(sensor_times):
        raise RuntimeError(
            f"Only {len(sensor_times) - recovery_index} samples remain after "
            f"recovery; requested {steps}"
        )

    corrections = []
    for index in range(recovery_index, alignment_end):
        corrections.append(
            pose_matrix(lio_position[index], lio_quaternion[index])
            @ np.linalg.inv(
                pose_matrix(glim_position[index], glim_quaternion[index])
            )
        )
    corrections = np.stack(corrections)
    correction = np.eye(4, dtype=np.float64)
    correction[:3, :3] = Rotation.from_matrix(
        corrections[:, :3, :3]
    ).mean().as_matrix()
    correction[:3, 3] = np.median(corrections[:, :3, 3], axis=0)

    recovery_elapsed = float(sensor_times[recovery_index] - common_start)
    alignment_end_elapsed = float(sensor_times[alignment_end - 1] - common_start)
    previous_note = glim.spatial_alignment_note
    note = (
        f"{previous_note}; then one fixed post-recovery correction was frozen "
        f"from {steps} samples at common elapsed "
        f"{recovery_elapsed:.3f}..{alignment_end_elapsed:.3f} s "
        f"(recovery={consecutive_steps} consecutive samples <= "
        f"{position_threshold:.3f} m, <= {orientation_threshold:.3f} deg). "
        "This removes the initial relative frame bias and evaluates tracking, "
        "not absolute global-localization error"
    )
    apply_spatial_transform(glim, correction, note)
    return correction, recovery_elapsed, alignment_end_elapsed


def compute_error_series(
    glim: Trajectory,
    lio: Trajectory,
    common_start: float,
    common_end: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate both estimators at every GLIM sensor timestamp in the overlap."""
    mask = (glim.times >= common_start) & (glim.times <= common_end)
    sensor_times = np.unique(glim.times[mask])
    if len(sensor_times) < 2:
        sensor_times = np.linspace(common_start, common_end, 2)
    glim_position, glim_quaternion = _sample_many(glim, sensor_times)
    lio_position, lio_quaternion = _sample_many(lio, sensor_times)
    # Keep the axis errors signed so a persistent map-frame bias remains
    # visible.  Positive means Super-LIO is positive of GLIM on that axis.
    position_component_error = lio_position - glim_position
    position_error = np.linalg.norm(position_component_error, axis=1)
    orientation_error = np.rad2deg(
        (
            Rotation.from_quat(lio_quaternion).inv()
            * Rotation.from_quat(glim_quaternion)
        ).magnitude()
    )
    return (
        sensor_times - common_start,
        position_error,
        position_component_error,
        orientation_error,
    )


def current_error(
    glim: Trajectory, lio: Trajectory, sensor_timestamp: float
) -> tuple[float, np.ndarray, float]:
    gp, gq, _ = glim.sample(sensor_timestamp)
    lp, lq, _ = lio.sample(sensor_timestamp)
    position_components = lp - gp
    position = float(np.linalg.norm(position_components))
    orientation = float(
        np.rad2deg(
            (Rotation.from_quat(lq).inv() * Rotation.from_quat(gq)).magnitude()
        )
    )
    return position, position_components, orientation


def error_figure(
    elapsed: np.ndarray,
    position_error: np.ndarray,
    position_component_error: np.ndarray,
) -> go.Figure:
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=(
            "3D position error norm [m]",
            "Signed x error: Super LIO − GLIM [m]",
            "Signed y error: Super LIO − GLIM [m]",
            "Signed z error: Super LIO − GLIM [m]",
        ),
    )
    values = (
        position_error,
        position_component_error[:, 0],
        position_component_error[:, 1],
        position_component_error[:, 2],
    )
    colors = ("#FFCA28", "#EF5350", "#66BB6A", "#42A5F5")
    names = ("3D norm", "x", "y", "z")
    for row, (value, color, name) in enumerate(
        zip(values, colors, names), start=1
    ):
        figure.add_trace(
            go.Scattergl(
                x=elapsed,
                y=value,
                name=name,
                showlegend=False,
                line={"color": color, "width": 1.7},
            ),
            row=row,
            col=1,
        )
    for row, (value, color) in enumerate(zip(values, colors), start=1):
        figure.add_trace(
            go.Scattergl(
                x=[0.0],
                y=[float(value[0])],
                mode="markers",
                showlegend=False,
                marker={
                    "color": color,
                    "size": 11,
                    "line": {"color": "#FFFFFF", "width": 2},
                },
            ),
            row=row,
            col=1,
        )
    figure.add_shape(
        type="line",
        x0=0.0,
        x1=0.0,
        y0=0.0,
        y1=1.0,
        xref="x",
        yref="paper",
        editable=False,
        line={"color": "#F5C542", "width": 5},
    )
    figure.update_layout(
        title={
            "text": "Same sensor timestamp · cursor follows slider/keyboard",
            "font": {"size": 15},
        },
        height=760,
        margin={"l": 48, "r": 12, "t": 48, "b": 38},
        hovermode="x unified",
        dragmode="zoom",
        meta={"g1_position_error_autoscale": True},
        uirevision="g1-position-error-axes",
        template="plotly_dark",
    )
    figure.update_xaxes(
        title_text="sensor time from common overlap start [s]", row=4, col=1
    )
    max_absolute_error = max(
        float(np.max(np.abs(value))) for value in values if len(value)
    )
    shared_limit = max(max_absolute_error * 1.12, 1.0e-4)
    figure.update_yaxes(
        tickformat=".3f",
        range=(-shared_limit, shared_limit),
        autorange=False,
        zeroline=True,
        zerolinecolor="#808080",
    )
    return figure


def update_error_figure(
    figure: go.Figure,
    elapsed: float,
    position_error: float,
    position_component_error: np.ndarray,
) -> None:
    _set_figure_cursor(figure, elapsed)
    current_values = (position_error, *position_component_error)
    for trace_index, value in enumerate(current_values, start=4):
        figure.data[trace_index].x = (elapsed,)
        figure.data[trace_index].y = (float(value),)


class DualReplayApp:
    def __init__(
        self,
        args: argparse.Namespace,
        bag: BagData,
        trajectories: list[Trajectory],
        clock_topic: str,
        glim_source: str,
        spatial_alignment: np.ndarray,
        map_points: np.ndarray,
        map_colors: np.ndarray,
    ) -> None:
        import viser

        self.args = args
        self.bag = bag
        self.trajectories = trajectories
        self.dual_mode = len(trajectories) == 2
        if len(trajectories) not in (1, 2):
            raise ValueError(f"Expected one or two trajectories, got {len(trajectories)}")
        self.lio = trajectories[-1]
        self.glim = trajectories[0] if self.dual_mode else None
        self.origin_trajectory = self.glim if self.glim is not None else self.lio
        self.clock_topic = clock_topic
        self.glim_source = glim_source
        self.spatial_alignment = spatial_alignment
        self.map_points = map_points
        self.common_start = float(max(item.times[0] for item in trajectories))
        self.common_end = float(min(item.times[-1] for item in trajectories))
        self.max_elapsed = float(self.common_end - self.common_start)
        lio_frame_mask = (
            (self.lio.times >= self.common_start)
            & (self.lio.times <= self.common_end)
        )
        self.frame_sensor_times = np.unique(self.lio.times[lio_frame_mask])
        if len(self.frame_sensor_times) == 0:
            raise RuntimeError("No /lio/odom frames exist in the common interval")
        self.frame_elapsed = self.frame_sensor_times - self.common_start
        self.error_elapsed = np.empty(0, dtype=np.float64)
        self.position_errors = np.empty(0, dtype=np.float64)
        self.position_component_errors = np.empty((0, 3), dtype=np.float64)
        self.orientation_errors = np.empty(0, dtype=np.float64)
        if self.glim is not None:
            (
                self.error_elapsed,
                self.position_errors,
                self.position_component_errors,
                self.orientation_errors,
            ) = compute_error_series(
                self.glim, self.lio, self.common_start, self.common_end
            )

        self.server = viser.ViserServer(
            host=args.host,
            port=args.port,
            label=(
                f"G1 {args.run_id} · GLIM vs Super LIO"
                if self.dual_mode
                else f"G1 {args.run_id} · Super LIO"
            ),
        )
        self.server.scene.set_up_direction("+z")
        self.lock = threading.RLock()
        self.playing = False
        self.last_wall_time = time.monotonic()
        self.last_plot_update = 0.0

        self.map_handle = None
        if len(map_points):
            self.map_handle = self.server.scene.add_point_cloud(
                "/map/dump7_xyzi_clean",
                points=map_points,
                colors=map_colors,
                point_size=args.map_point_size,
                point_shape="circle",
                precision="float32",
            )
        self.pose_frames = {}
        self.pose_markers = {}
        self.trajectory_lines = {}
        for index, trajectory in enumerate(trajectories):
            root = f"/trajectories/trajectory_{index}"
            points = trajectory.positions.astype(np.float32)
            self.trajectory_lines[trajectory.name] = self.server.scene.add_line_segments(
                root + "/path",
                points=np.stack((points[:-1], points[1:]), axis=1),
                colors=trajectory.spec.color,
                line_width=3.0,
            )
            self.pose_frames[trajectory.name] = self.server.scene.add_frame(
                root + "/imu_pose", axes_length=0.22, axes_radius=0.008
            )
            self.pose_markers[trajectory.name] = self.server.scene.add_icosphere(
                root + "/current_point",
                radius=0.075,
                color=trajectory.spec.color,
            )
            self.server.scene.add_label(
                root + "/imu_pose/label",
                text=trajectory.name,
                position=(0.0, 0.0, 0.18),
            )

        self.robot_bases = {}
        self.robot_models = {}
        self.robot_cfg_indices: dict[str, np.ndarray] = {}
        self.kinematics_urdf = None
        if not args.no_robot:
            self._load_robots(args.urdf)

        self.scene_origin = self._initial_scene_origin()
        self.server.scene.add_grid(
            "/map/ground_grid",
            width=20.0,
            height=20.0,
            plane="xy",
            cell_size=0.5,
            cell_thickness=0.5,
            section_size=2.0,
            section_thickness=1.0,
            position=tuple(self.scene_origin),
        )
        self.server.scene.add_frame(
            "/map/replay_origin",
            position=tuple(self.scene_origin),
            axes_length=0.35,
            axes_radius=0.008,
        )
        print(
            f"Viser grid/origin placed below the initial "
            f"{self.origin_trajectory.name} robot feet at "
            f"{np.array2string(self.scene_origin, precision=4)}.",
            flush=True,
        )

        self.bag_aug_handle = None
        self.bag_aug_last_index = None
        self.cloud_world_handle = None
        self.cloud_world_last_index = None
        self.live_aug_handle = None
        self.live_aug_version = 0
        self.live_aug_receiver = None
        if not args.no_live_aug_map:
            self.live_aug_receiver = LiveAugmentedMapReceiver(
                args.live_aug_map_topic, args.max_aug_map_points
            )

        self._error_autoscale_bridge_source = None
        self._build_gui()
        self._bridge_sources = [self._keyboard_bridge_source]
        if self._error_autoscale_bridge_source is not None:
            self._bridge_sources.append(self._error_autoscale_bridge_source)

        # Viser's public client callback waits for the first camera packet.
        # Re-send browser controls after UI hydration so keyboard stepping works
        # reliably even on plot-heavy pages and with --no-plots.
        @self.server._websock_server.on_client_connect
        async def _(_connection) -> None:
            from viser import _messages as viser_messages

            async def _send_after_hydration() -> None:
                await asyncio.sleep(5.0)
                for source in self._bridge_sources:
                    _connection.queue_message(
                        viser_messages.RunJavascriptMessage(
                            source=source
                        )
                    )

            asyncio.create_task(_send_after_hydration())
        camera_points = map_points if len(map_points) else np.vstack(
            [item.positions for item in trajectories]
        )
        self._register_client_camera(camera_points, self.scene_origin)
        self.update_state(0.0, force_plots=True)

    def _initial_scene_origin(self) -> np.ndarray:
        """Return a visual-only origin below the initial reference robot feet."""
        initial_position, _, _ = self.origin_trajectory.sample(self.common_start)
        fallback = np.asarray(initial_position, dtype=np.float64).copy()
        fallback[2] -= 1.0
        if self.kinematics_urdf is None:
            return fallback

        record_timestamp = float(
            self.bag.record_time_from_sensor_time(
                self.common_start, self.clock_topic
            )
        )
        q, _, _, _, _, _ = self.bag.sample_lowstate(record_timestamp)
        imu_position, imu_quaternion, _ = self.origin_trajectory.sample(
            self.common_start
        )
        pelvis_position, pelvis_quaternion = self._pelvis_pose(
            imu_position,
            imu_quaternion,
            q,
            record_timestamp,
        )
        map_from_pelvis = pose_matrix(pelvis_position, pelvis_quaternion)
        sole_points = []
        for link in ("left_ankle_roll_link", "right_ankle_roll_link"):
            pelvis_from_ankle = self.kinematics_urdf.get_transform(
                link, "pelvis"
            )
            sole_local = np.array(
                (0.0, 0.0, ANKLE_ROLL_SOLE_Z, 1.0), dtype=np.float64
            )
            sole_points.append(
                (map_from_pelvis @ pelvis_from_ankle @ sole_local)[:3]
            )
        sole_points = np.stack(sole_points)
        return np.array(
            (
                float(np.mean(sole_points[:, 0])),
                float(np.mean(sole_points[:, 1])),
                float(np.min(sole_points[:, 2])),
            ),
            dtype=np.float64,
        )

    def _load_robots(self, urdf_path: Path) -> None:
        import yourdfpy
        from viser.extras import ViserUrdf

        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        print(
            f"Loading {len(self.trajectories)} translucent G1 robot "
            f"hypothes{'is' if len(self.trajectories) == 1 else 'es'}...",
            flush=True,
        )
        visual_urdf = yourdfpy.URDF.load(
            urdf_path,
            build_scene_graph=True,
            build_collision_scene_graph=False,
            load_meshes=True,
            load_collision_meshes=False,
            filename_handler=partial(
                yourdfpy.filename_handler_magic, dir=urdf_path.parent
            ),
        )
        self.kinematics_urdf = yourdfpy.URDF.load(
            urdf_path,
            build_scene_graph=True,
            build_collision_scene_graph=False,
            load_meshes=False,
            load_collision_meshes=False,
            filename_handler=partial(
                yourdfpy.filename_handler_magic, dir=urdf_path.parent
            ),
        )
        for index, trajectory in enumerate(self.trajectories):
            color = tuple(channel / 255.0 for channel in trajectory.spec.color) + (
                0.66,
            )
            root = f"/robots/robot_{index}"
            self.robot_bases[trajectory.name] = self.server.scene.add_frame(
                root, show_axes=False
            )
            model = ViserUrdf(
                self.server,
                visual_urdf,
                root_node_name=root,
                mesh_color_override=color,
            )
            self.robot_models[trajectory.name] = model
            joint_names = model.get_actuated_joint_names()
            name_to_index = {name: i for i, name in enumerate(joint_names)}
            self.robot_cfg_indices[trajectory.name] = np.array(
                [name_to_index[name] for name in G1_JOINT_NAMES], dtype=np.int64
            )

    def _pelvis_pose(
        self,
        imu_position: np.ndarray,
        imu_quaternion: np.ndarray,
        q: np.ndarray,
        record_timestamp: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct T_map_pelvis from the physical Livox/URDF frame chain."""
        map_from_world = _required_tf(
            self.bag, "map", "world", record_timestamp
        )
        world_from_imu = pose_matrix(imu_position, imu_quaternion)
        frame_from_imu = _required_tf(
            self.bag,
            "mid360_link_frame",
            "mid360_link_IMU",
            record_timestamp,
        )
        mid360_from_frame = _required_tf(
            self.bag,
            "mid360_link",
            "mid360_link_frame",
            record_timestamp,
        )
        if self.kinematics_urdf is None:
            raise RuntimeError("Robot kinematics URDF is not loaded")
        self.kinematics_urdf.update_cfg(dict(zip(G1_JOINT_NAMES, q)))

        # yourdfpy.get_transform(frame_to, frame_from) returns
        # T_frame_from_frame_to.  Therefore this is T_mid360_link_pelvis.
        mid360_from_pelvis = self.kinematics_urdf.get_transform(
            "pelvis", "mid360_link"
        )

        # /lio/odom is T_world_mid360_link_IMU.  The bag's imu->lidar alias
        # has the translation sign reversed, so compose through the named
        # physical frames instead:
        # T_map_world * T_world_imu * T_imu_frame * T_frame_mid360
        #             * T_mid360_pelvis.
        imu_from_frame = np.linalg.inv(frame_from_imu)
        frame_from_mid360 = np.linalg.inv(mid360_from_frame)
        return matrix_pose(
            map_from_world
            @ world_from_imu
            @ imu_from_frame
            @ frame_from_mid360
            @ mid360_from_pelvis
        )

    def _replace_cloud(
        self,
        handle,
        name: str,
        points: np.ndarray,
        color: tuple[int, int, int],
        visible: bool,
    ):
        if handle is not None:
            handle.points = np.asarray(points, dtype=np.float32)
            handle.colors = np.asarray(color, dtype=np.uint8)
            handle.visible = visible
            return handle
        return self.server.scene.add_point_cloud(
            name,
            points=points,
            colors=color,
            point_size=float(self.aug_map_point_size.value),
            point_shape="circle",
            precision="float32",
            visible=visible,
        )

    def _update_bag_augmented_map(self, elapsed: float) -> None:
        if not bool(self.show_bag_aug_map.value):
            if self.bag_aug_handle is not None:
                self.bag_aug_handle.visible = False
            return
        series = self.bag.pointclouds.get("/augmented_map")
        if series is None:
            self.bag_aug_status.value = "topic not present"
            return
        sensor_timestamp = self.common_start + elapsed
        record_timestamp = float(
            self.bag.record_time_from_sensor_time(
                sensor_timestamp, self.clock_topic
            )
        )
        index = series.frame_index_at(record_timestamp)
        if index is None:
            if self.bag_aug_handle is not None:
                self.bag_aug_handle.visible = False
            self.bag_aug_status.value = "waiting for first frame"
            return
        if index != self.bag_aug_last_index or self.bag_aug_handle is None:
            frame = series.load_frame(
                index, max_points=self.args.max_aug_map_points
            )
            self.bag_aug_handle = self._replace_cloud(
                self.bag_aug_handle,
                f"/augmented_map/bag_{self.args.run_id}",
                frame.points,
                (255, 170, 55),
                True,
            )
            self.bag_aug_last_index = index
            self.bag_aug_status.value = (
                f"frame {index + 1}/{len(series.times)}, {len(frame.points):,} points, "
                f"frame_id={frame.frame_id or '<empty>'}"
            )
        else:
            self.bag_aug_handle.visible = True

    def _update_cloud_world(self, elapsed: float) -> None:
        """Replay the newest causal /lio/cloud_world frame at the slider time."""
        if not bool(self.show_cloud_world.value):
            if self.cloud_world_handle is not None:
                self.cloud_world_handle.visible = False
            return
        series = self.bag.pointclouds.get("/lio/cloud_world")
        if series is None:
            self.cloud_world_status.value = "topic not present"
            return
        sensor_timestamp = self.common_start + elapsed
        record_timestamp = float(
            self.bag.record_time_from_sensor_time(
                sensor_timestamp, self.clock_topic
            )
        )
        index = series.frame_index_at(record_timestamp)
        if index is None:
            if self.cloud_world_handle is not None:
                self.cloud_world_handle.visible = False
            self.cloud_world_status.value = "waiting for first frame"
            return
        if index != self.cloud_world_last_index or self.cloud_world_handle is None:
            frame = series.load_frame(
                index, max_points=self.args.max_cloud_world_points
            )
            points = frame.points
            normalized_frame = frame.frame_id.strip().lstrip("/")
            if normalized_frame == "world":
                map_from_world = _required_tf(
                    self.bag, "map", "world", record_timestamp
                )
                points = (
                    Rotation.from_matrix(map_from_world[:3, :3]).apply(points)
                    + map_from_world[:3, 3]
                ).astype(np.float32)
            self.cloud_world_handle = self._replace_cloud(
                self.cloud_world_handle,
                f"/lio/cloud_world/bag_{self.args.run_id}",
                points,
                (70, 180, 255),
                True,
            )
            self.cloud_world_last_index = index
            self.cloud_world_status.value = (
                f"frame {index + 1}/{len(series.times)}, {len(points):,} points, "
                f"frame_id={frame.frame_id or '<empty>'}"
            )
        else:
            self.cloud_world_handle.visible = True

    def _update_live_augmented_map(self) -> None:
        receiver = self.live_aug_receiver
        if receiver is None:
            return
        if receiver.error is not None:
            self.live_aug_status.value = receiver.error
        snapshot = receiver.snapshot(self.live_aug_version)
        if snapshot is None:
            return
        version, points, stamp, frame_id = snapshot
        self.live_aug_version = version
        self.live_aug_handle = self._replace_cloud(
            self.live_aug_handle,
            "/augmented_map/live",
            points,
            (245, 80, 225),
            bool(self.show_live_aug_map.value),
        )
        self.live_aug_status.value = (
            f"message {version}, {len(points):,} points, stamp={stamp:.6f}, "
            f"frame_id={frame_id or '<empty>'}"
        )

    def _build_gui(self) -> None:
        gui = self.server.gui
        run_time = _format_run_time(self.args.run_id)
        if self.dual_mode:
            gui.add_markdown(
                f"# {run_time} · GLIM vs Super LIO\n"
                "두 IMU pose를 **같은 절대 sensor timestamp**에 보간해 비교합니다. "
                "GLIM=빨강, Super LIO=초록. 두 반투명 G1은 같은 `/lowstate`를 "
                "사용하며 localization pose만 다릅니다."
            )
        else:
            gui.add_markdown(
                f"# {run_time} · Super LIO replay\n"
                "이 bag의 `/lio/odom` pose, `/lowstate`, TF와 "
                "`/augmented_map`만 동기 재생합니다. GLIM 데이터나 임의 정렬은 "
                "사용하지 않습니다."
            )
        with gui.add_folder("Data source audit", expand_by_default=False):
            audit = (
                f"- evaluated LIO bag: `{self.args.bag}`\n"
                f"- replay context bag: `{self.bag.path}`\n"
                f"- LIO: `{self.clock_topic}` (`header.stamp`)\n"
                f"- LIO filtering: `{self.lio.filter_note}`"
            )
            if self.dual_mode:
                audit += (
                    f"\n- GLIM: `{self.glim_source}`\n\n"
                    "bag에 `/glim_ros/odom_corrected`가 없으면 동일 run의 "
                    "`traj_imu.txt`를 사용합니다."
                )
            gui.add_markdown(audit)
        if self.glim is not None:
            with gui.add_folder(
                "Fixed start-frame anchor", expand_by_default=False
            ):
                gui.add_markdown(
                    "같은 run의 원본 bag과 GLIM dump의 첫 대응 pose에서 계산한 "
                    "SE(3) **하나만** 모든 GLIM pose에 동일하게 왼쪽 곱합니다. "
                    "원본 bag의 `/odometry/odom`은 recorded live Super-LIO pose에 "
                    "IMU→LiDAR extrinsic을 곱한 값입니다. 따라서 이는 독립 측량 "
                    "calibration이 아니라 **초기 frame anchor**이며 trajectory "
                    "fitting은 하지 않습니다.\n\n"
                    "`T_MAP_GLIM = T_MAP_LIDAR(/odometry/odom) · "
                    "inverse(T_GLIM_LIDAR(traj_lidar.txt))`\n\n"
                    f"{self.glim.spatial_alignment_note}\n\n"
                    "```text\n"
                    + np.array2string(
                        self.spatial_alignment, precision=6, suppress_small=True
                    )
                    + "\n```"
                )
        with gui.add_folder("Playback", expand_by_default=True):
            self.time_slider = gui.add_slider(
                "common sensor timestamp offset [s]",
                min=0.0,
                max=self.max_elapsed,
                step=0.001,
                initial_value=0.0,
                marks=(
                    (0.0, "0"),
                    (round(self.max_elapsed, 3), f"{self.max_elapsed:.1f}s"),
                ),
            )
            self.play_button = gui.add_button(
                "Play / Pause  [Space]", icon="player-play"
            )
            self.previous_frame_button = gui.add_button(
                "Previous /lio/odom frame  [, or ←]"
            )
            self.next_frame_button = gui.add_button(
                "Next /lio/odom frame  [. or →]"
            )
            self.current_frame_number = gui.add_number(
                "current /lio/odom frame",
                1,
                step=1,
                disabled=True,
            )
            self.frame_count_text = gui.add_text(
                "frame stepping",
                f"1 / {len(self.frame_sensor_times):,} · ,/. step · Space toggle",
                disabled=True,
            )
            self.speed_control = gui.add_button_group(
                "speed", options=("0.25×", "0.5×", "1×", "2×")
            )
            self.speed_control.value = "1×"
            self.elapsed_number = gui.add_number(
                "current elapsed [s]", 0.0, step=0.001, disabled=True
            )
            self.sensor_timestamp = gui.add_number(
                "sensor timestamp", self.common_start, disabled=True
            )
            self.record_timestamp = gui.add_number(
                "bag record timestamp",
                float(
                    self.bag.record_time_from_sensor_time(
                        self.common_start, self.clock_topic
                    )
                ),
                disabled=True,
            )

        with gui.add_folder("Visibility", expand_by_default=False):
            gui.add_markdown(
                "G1 pose uses only `/lio/odom`, named physical sensor TFs, "
                "URDF FK, and `/lowstate`; no floor fitting or z-offset."
            )
            self.show_map = gui.add_checkbox(
                "static PCD map",
                self.map_handle is not None,
                disabled=self.map_handle is None,
            )
            self.show_paths = gui.add_checkbox("full trajectories", True)
            self.show_frames = gui.add_checkbox("current IMU frames", True)
            self.show_robot = {}
            with gui.add_folder("G1 robot meshes", expand_by_default=True):
                for trajectory in self.trajectories:
                    self.show_robot[trajectory.name] = gui.add_checkbox(
                        trajectory.name,
                        not self.args.no_robot,
                        disabled=self.args.no_robot,
                    )

        with gui.add_folder(
            "Point clouds / augmented maps", expand_by_default=True
        ):
            self.aug_map_point_size = gui.add_slider(
                "all point cloud size",
                min=0.005,
                max=0.08,
                step=0.005,
                initial_value=float(self.args.aug_map_point_size),
            )
            bag_available = "/augmented_map" in self.bag.pointclouds
            self.show_bag_aug_map = gui.add_checkbox(
                "bag /augmented_map", bag_available, disabled=not bag_available
            )
            self.bag_aug_status = gui.add_text(
                "bag map status",
                "waiting for first frame" if bag_available else "topic not present",
                disabled=True,
            )
            cloud_world_available = "/lio/cloud_world" in self.bag.pointclouds
            self.show_cloud_world = gui.add_checkbox(
                "bag /lio/cloud_world",
                cloud_world_available,
                disabled=not cloud_world_available,
            )
            self.cloud_world_status = gui.add_text(
                "cloud_world status",
                (
                    "waiting for first frame"
                    if cloud_world_available
                    else "topic not present"
                ),
                disabled=True,
            )
            live_available = (
                self.live_aug_receiver is not None
                and self.live_aug_receiver.error is None
            )
            self.show_live_aug_map = gui.add_checkbox(
                f"live {self.args.live_aug_map_topic}",
                live_available,
                disabled=not live_available,
            )
            live_status = "waiting for ROS message"
            if self.args.no_live_aug_map:
                live_status = "disabled by --no-live-aug-map"
            elif self.live_aug_receiver is not None and self.live_aug_receiver.error:
                live_status = self.live_aug_receiver.error
            self.live_aug_status = gui.add_text(
                "live map status", live_status, disabled=True
            )

        self.pose_gui = {}
        with gui.add_folder("Current trajectory values", expand_by_default=True):
            for trajectory in self.trajectories:
                with gui.add_folder(trajectory.name, expand_by_default=False):
                    position = gui.add_vector3(
                        "position xyz [m]",
                        (0.0, 0.0, 0.0),
                        step=0.001,
                        disabled=True,
                    )
                    rpy = gui.add_vector3(
                        "orientation rpy [deg]",
                        (0.0, 0.0, 0.0),
                        step=0.01,
                        disabled=True,
                    )
                    row = gui.add_number("nearest source row", 0, disabled=True)
                    nearest_dt = gui.add_number(
                        "nearest row dt [ms]", 0.0, step=0.001, disabled=True
                    )
                    self.pose_gui[trajectory.name] = (
                        position,
                        rpy,
                        row,
                        nearest_dt,
                    )

        with gui.add_folder("LowState joint inspector", expand_by_default=True):
            self.joint_name = gui.add_dropdown(
                "joint",
                options=G1_JOINT_NAMES,
                initial_value="left_hip_pitch_joint",
            )
            self.current_q = gui.add_number(
                "q [rad]", 0.0, step=0.001, disabled=True
            )
            self.current_dq = gui.add_number(
                "dq [rad/s]", 0.0, step=0.001, disabled=True
            )
            self.current_tau = gui.add_number(
                "tau_est [Nm]", 0.0, step=0.001, disabled=True
            )
            self.current_imu_rpy = gui.add_vector3(
                "lowstate IMU rpy [rad]",
                (0.0, 0.0, 0.0),
                step=0.001,
                disabled=True,
            )

        self.error_figure = None
        self.error_plot_handle = None
        self.pose_figures = {}
        self.pose_plot_handles = {}
        self.joint_figure = None
        self.joint_plot_handle = None
        self.current_position_error = None
        self.current_position_component_errors = []
        self.current_orientation_error = None
        self.error_time_slider = None
        if self.dual_mode:
            with gui.add_folder("Same-timestamp error", expand_by_default=True):
                gui.add_markdown(
                    "아래 graph timeline 또는 키보드 `,`/`.` (`[`/`]`)로 "
                    "시간을 이동합니다. 축 오차는 `Super LIO − GLIM`의 "
                    "signed 값입니다."
                )
                if self.args.post_recovery_align_steps > 0:
                    gui.add_markdown(
                        "정렬 모드: startup recovery 감지 후 "
                        f"`N={self.args.post_recovery_align_steps}` pose의 "
                        "SE(3) 평균을 한 번 계산해 고정했습니다. 이 그래프는 "
                        "초기 절대오차가 아니라 recovery 이후 tracking 오차를 "
                        "평가합니다."
                    )
                self.current_position_error = gui.add_number(
                    "3D position error [m]", 0.0, step=0.0001, disabled=True
                )
                self.current_position_component_errors = [
                    gui.add_number(
                        f"{axis} error [m]", 0.0, step=0.0001, disabled=True
                    )
                    for axis in ("x", "y", "z")
                ]
                self.current_orientation_error = gui.add_number(
                    "orientation error [deg]", 0.0, step=0.001, disabled=True
                )
                if not self.args.no_plots:
                    self.error_time_slider = gui.add_slider(
                        "graph timeline [s]",
                        min=0.0,
                        max=self.max_elapsed,
                        step=0.001,
                        initial_value=0.0,
                        marks=(
                            (0.0, "0"),
                            (
                                round(self.max_elapsed, 3),
                                f"{self.max_elapsed:.1f}s",
                            ),
                        ),
                    )
                    self.error_figure = error_figure(
                        self.error_elapsed,
                        self.position_errors,
                        self.position_component_errors,
                    )
                    self.error_plot_handle = gui.add_plotly(
                        self.error_figure,
                        aspect=1.05,
                        config={
                            "displaylogo": False,
                            "responsive": True,
                            "editable": False,
                            "scrollZoom": True,
                        },
                    )

        if not self.args.no_plots:
            with gui.add_folder("Trajectory plots", expand_by_default=False):
                for trajectory in self.trajectories:
                    figure = _pose_figure(trajectory, self.common_start)
                    self.pose_figures[trajectory.name] = figure
                    self.pose_plot_handles[trajectory.name] = gui.add_plotly(
                        figure,
                        aspect=1.75,
                        config={"displaylogo": False, "responsive": True},
                    )
            with gui.add_folder("Selected joint plot", expand_by_default=False):
                self.joint_figure = _joint_figure(
                    self.bag,
                    0,
                    0.0,
                    self.common_start,
                    self.clock_topic,
                )
                self.joint_plot_handle = gui.add_plotly(
                    self.joint_figure,
                    aspect=1.5,
                    config={"displaylogo": False, "responsive": True},
                )

        @self.time_slider.on_update
        def _(_) -> None:
            elapsed = float(self.time_slider.value)
            if (
                self.error_time_slider is not None
                and abs(float(self.error_time_slider.value) - elapsed) > 1.0e-9
            ):
                self.error_time_slider.value = elapsed
            self.update_state(elapsed)

        if self.error_time_slider is not None:
            @self.error_time_slider.on_update
            def _(_) -> None:
                elapsed = float(self.error_time_slider.value)
                if abs(float(self.time_slider.value) - elapsed) > 1.0e-9:
                    self.time_slider.value = elapsed
                self.update_state(elapsed)

        @self.play_button.on_click
        def _(_) -> None:
            self._toggle_playback()

        @self.previous_frame_button.on_click
        def _(_) -> None:
            self._step_frame(-1)

        @self.next_frame_button.on_click
        def _(_) -> None:
            self._step_frame(1)

        @self.show_map.on_update
        def _(_) -> None:
            if self.map_handle is not None:
                self.map_handle.visible = bool(self.show_map.value)

        @self.show_paths.on_update
        def _(_) -> None:
            for handle in self.trajectory_lines.values():
                handle.visible = bool(self.show_paths.value)

        @self.show_frames.on_update
        def _(_) -> None:
            self.update_state(float(self.time_slider.value))

        for trajectory_name, control in self.show_robot.items():
            @control.on_update
            def _(_, name=trajectory_name, checkbox=control) -> None:
                if name in self.robot_bases:
                    self.robot_bases[name].visible = bool(checkbox.value)

        @self.show_bag_aug_map.on_update
        def _(_) -> None:
            self._update_bag_augmented_map(float(self.time_slider.value))

        @self.show_cloud_world.on_update
        def _(_) -> None:
            self._update_cloud_world(float(self.time_slider.value))

        @self.show_live_aug_map.on_update
        def _(_) -> None:
            if self.live_aug_handle is not None:
                self.live_aug_handle.visible = bool(self.show_live_aug_map.value)

        @self.aug_map_point_size.on_update
        def _(_) -> None:
            value = float(self.aug_map_point_size.value)
            if self.map_handle is not None:
                self.map_handle.point_size = value
            if self.bag_aug_handle is not None:
                self.bag_aug_handle.point_size = value
            if self.cloud_world_handle is not None:
                self.cloud_world_handle.point_size = value
            if self.live_aug_handle is not None:
                self.live_aug_handle.point_size = value

        @self.joint_name.on_update
        def _(_) -> None:
            self._rebuild_joint_plot(float(self.time_slider.value))
            self.update_state(float(self.time_slider.value), force_plots=True)

        self._install_keyboard_bridge()
        if self.dual_mode and not self.args.no_plots:
            self._install_error_autoscale_bridge()

    def _install_keyboard_bridge(self) -> None:
        """Map Space/arrow keys to the synchronized Viser playback controls."""
        from viser import _messages as viser_messages

        play_button_uuid = self.play_button._impl.uuid
        previous_button_uuid = self.previous_frame_button._impl.uuid
        next_button_uuid = self.next_frame_button._impl.uuid
        script = f"""
(() => {{
  if (window.__g1ReplayKeyboardHandler) {{
    window.removeEventListener(
      'keydown', window.__g1ReplayKeyboardHandler, true
    );
  }}
  const playButtonId = {play_button_uuid!r};
  const previousButtonId = {previous_button_uuid!r};
  const nextButtonId = {next_button_uuid!r};

  function clickGuiButton(elementId) {{
    const root = document.getElementById(elementId);
    if (!root) return false;
    const button = root.matches && root.matches('button')
      ? root
      : (root.querySelector && root.querySelector('button')) ||
        (root.parentElement && root.parentElement.querySelector('button'));
    const target = button || root;
    target.dispatchEvent(new MouseEvent('click', {{
      bubbles: true,
      cancelable: true,
      view: window,
    }}));
    return true;
  }}

  function handler(event) {{
    const target = event.target;
    if (target && target.closest &&
        target.closest('input, textarea, select, [contenteditable="true"]')) {{
      return;
    }}
    let handled = false;
    if (event.code === 'Space') {{
      handled = clickGuiButton(playButtonId);
    }} else if (event.code === 'Comma' ||
               event.code === 'BracketLeft' ||
               event.code === 'ArrowLeft') {{
      handled = clickGuiButton(previousButtonId);
    }} else if (event.code === 'Period' ||
               event.code === 'BracketRight' ||
               event.code === 'ArrowRight') {{
      handled = clickGuiButton(nextButtonId);
    }}
    if (handled) {{
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
    }}
  }}

  window.__g1ReplayKeyboardHandler = handler;
  window.addEventListener('keydown', handler, true);
}})();
"""
        self._keyboard_bridge_source = script
        self.server.gui._websock_interface.queue_message(
            viser_messages.RunJavascriptMessage(source=script)
        )

    def _install_error_autoscale_bridge(self) -> None:
        """Give every error subplot one shared scale for the visible interval."""
        from viser import _messages as viser_messages

        script = r"""
(() => {
  if (window.__g1ErrorAutoscaleBridge) {
    clearInterval(window.__g1ErrorAutoscaleBridge);
  }

  function visibleXRange(plot) {
    const layout = plot._fullLayout;
    if (!layout) return null;
    for (const name of ['xaxis4', 'xaxis3', 'xaxis2', 'xaxis']) {
      const axis = layout[name];
      if (axis && Array.isArray(axis.range) && axis.range.length === 2) {
        const left = Number(axis.range[0]);
        const right = Number(axis.range[1]);
        if (Number.isFinite(left) && Number.isFinite(right)) {
          return left <= right ? [left, right] : [right, left];
        }
      }
    }
    return null;
  }

  function autoscaleVisibleY(plot) {
    if (plot.__g1AutoscalingY || !window.Plotly) return;
    const xRange = visibleXRange(plot);
    if (!xRange) return;

    let maxAbsoluteError = 0.0;
    let foundValue = false;
    for (let traceIndex = 0; traceIndex < 4; ++traceIndex) {
      const trace = plot.data && plot.data[traceIndex];
      if (!trace || !trace.x || !trace.y) continue;
      const count = Math.min(trace.x.length, trace.y.length);
      for (let i = 0; i < count; ++i) {
        const x = Number(trace.x[i]);
        const y = Number(trace.y[i]);
        if (Number.isFinite(x) && Number.isFinite(y) &&
            x >= xRange[0] && x <= xRange[1]) {
          maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(y));
          foundValue = true;
        }
      }
    }
    if (!foundValue) return;

    const sharedLimit = Math.max(maxAbsoluteError * 1.12, 1e-4);
    const sharedRange = [-sharedLimit, sharedLimit];
    const relayout = {};
    for (let traceIndex = 0; traceIndex < 4; ++traceIndex) {
      const axisName = traceIndex === 0 ? 'yaxis' : `yaxis${traceIndex + 1}`;
      relayout[`${axisName}.range`] = sharedRange;
      relayout[`${axisName}.autorange`] = false;
    }

    plot.__g1AutoscalingY = true;
    Promise.resolve(window.Plotly.relayout(plot, relayout)).finally(() => {
      plot.__g1AutoscalingY = false;
    });
  }

  function axisRangeChanged(event) {
    return Object.keys(event || {}).some((key) =>
      /^[xy]axis\d*\.(range(\[\d\])?|autorange)$/.test(key)
    );
  }

  function attach() {
    document.querySelectorAll('.js-plotly-plot').forEach((plot) => {
      if (!plot._fullLayout || !plot._fullLayout.meta ||
          !plot._fullLayout.meta.g1_position_error_autoscale ||
          plot.__g1ErrorAutoscaleAttached) return;
      plot.__g1ErrorAutoscaleAttached = true;
      plot.on('plotly_relayout', (event) => {
        if (plot.__g1AutoscalingY || !axisRangeChanged(event)) return;
        window.requestAnimationFrame(() => autoscaleVisibleY(plot));
      });
      window.requestAnimationFrame(() => autoscaleVisibleY(plot));
    });
  }

  window.__g1ErrorAutoscaleBridge = setInterval(attach, 250);
  attach();
})();
"""
        self._error_autoscale_bridge_source = script
        self.server.gui._websock_interface.queue_message(
            viser_messages.RunJavascriptMessage(source=script)
        )

    def _rebuild_joint_plot(self, elapsed: float) -> None:
        if self.joint_plot_handle is None:
            return
        index = G1_JOINT_NAMES.index(str(self.joint_name.value))
        self.joint_figure = _joint_figure(
            self.bag,
            index,
            elapsed,
            self.common_start,
            self.clock_topic,
        )
        self.joint_plot_handle.figure = self.joint_figure

    def _nearest_frame_index(self, elapsed: float) -> int:
        right = int(np.searchsorted(self.frame_elapsed, elapsed, side="left"))
        if right <= 0:
            return 0
        if right >= len(self.frame_elapsed):
            return len(self.frame_elapsed) - 1
        left = right - 1
        return (
            left
            if elapsed - self.frame_elapsed[left]
            <= self.frame_elapsed[right] - elapsed
            else right
        )

    def _toggle_playback(self) -> None:
        with self.lock:
            self.playing = not self.playing
            self.last_wall_time = time.monotonic()

    def _step_frame(self, direction: int) -> None:
        """Pause and move to the previous/next exact /lio/odom timestamp."""
        with self.lock:
            current = float(self.time_slider.value)
            epsilon = 1.0e-7
            if direction < 0:
                index = int(
                    np.searchsorted(
                        self.frame_elapsed, current - epsilon, side="left"
                    )
                    - 1
                )
            else:
                index = int(
                    np.searchsorted(
                        self.frame_elapsed, current + epsilon, side="right"
                    )
                )
            index = int(np.clip(index, 0, len(self.frame_elapsed) - 1))
            elapsed = float(self.frame_elapsed[index])
            self.playing = False
            self.last_wall_time = time.monotonic()
            self.time_slider.value = elapsed
            self.update_state(elapsed, force_plots=True)

    def _register_client_camera(
        self, points: np.ndarray, scene_origin: np.ndarray
    ) -> None:
        low = np.percentile(points, 2.0, axis=0)
        high = np.percentile(points, 98.0, axis=0)
        radius = max(float(np.linalg.norm(high - low)), 4.0)
        distance = float(np.clip(radius * 0.25, 3.0, 7.0))
        camera_target = np.asarray(scene_origin, dtype=np.float64) + np.array(
            (0.0, 0.0, 0.75)
        )

        @self.server.on_client_connect
        def _(client) -> None:
            # Also send through the per-client socket once the camera handshake is
            # complete. This is intentionally not a broadcast/persistent message.
            from viser import _messages as viser_messages
            for source in self._bridge_sources:
                client._websock_connection.queue_message(
                    viser_messages.RunJavascriptMessage(
                        source=source
                    )
                )
            client.camera.up_direction = (0.0, 0.0, 1.0)
            client.camera.look_at = camera_target
            client.camera.position = camera_target + np.array(
                (0.65 * distance, -distance, 0.55 * distance)
            )

    def update_state(self, elapsed: float, force_plots: bool = False) -> None:
        with self.lock:
            elapsed = float(np.clip(elapsed, 0.0, self.max_elapsed))
            if (
                self.error_time_slider is not None
                and abs(float(self.error_time_slider.value) - elapsed) > 1.0e-9
            ):
                # Keep the graph-adjacent timeline moving during playback and
                # keyboard frame stepping as well as direct slider dragging.
                self.error_time_slider.value = elapsed
            sensor_timestamp = self.common_start + elapsed
            record_timestamp = float(
                self.bag.record_time_from_sensor_time(
                    sensor_timestamp, self.clock_topic
                )
            )
            self.elapsed_number.value = round(elapsed, 4)
            self.sensor_timestamp.value = sensor_timestamp
            self.record_timestamp.value = record_timestamp
            frame_index = self._nearest_frame_index(elapsed)
            frame_delta_ms = float(
                (self.frame_elapsed[frame_index] - elapsed) * 1.0e3
            )
            self.current_frame_number.value = frame_index + 1
            exact_label = (
                "exact timestamp"
                if abs(frame_delta_ms) < 1.0e-4
                else f"nearest Δ={frame_delta_ms:+.3f} ms"
            )
            self.frame_count_text.value = (
                f"{frame_index + 1:,} / {len(self.frame_sensor_times):,} · "
                f"{exact_label} · ,/. step · Space toggle"
            )

            q, dq, tau, imu_rpy, _, _ = self.bag.sample_lowstate(
                record_timestamp
            )
            pelvis_poses = {}
            for trajectory in self.trajectories:
                position, quaternion, source_index = trajectory.sample(
                    sensor_timestamp
                )
                frame = self.pose_frames[trajectory.name]
                marker = self.pose_markers[trajectory.name]
                frame.position = position
                frame.wxyz = quaternion_xyzw_to_wxyz(quaternion)
                frame.visible = bool(self.show_frames.value)
                marker.position = position
                marker.visible = bool(self.show_frames.value)

                position_gui, rpy_gui, row_gui, dt_gui = self.pose_gui[
                    trajectory.name
                ]
                position_gui.value = tuple(float(value) for value in position)
                rpy_gui.value = tuple(
                    float(value)
                    for value in Rotation.from_quat(quaternion).as_euler(
                        "xyz", degrees=True
                    )
                )
                row_gui.value = int(source_index + 1)
                dt_gui.value = float(
                    (trajectory.times[source_index] - sensor_timestamp) * 1e3
                )

                if trajectory.name in self.robot_models:
                    pelvis_poses[trajectory.name] = self._pelvis_pose(
                        position,
                        quaternion,
                        q,
                        record_timestamp,
                    )

            if self.robot_models:
                for trajectory in self.trajectories:
                    name = trajectory.name
                    base = self.robot_bases[name]
                    base.visible = bool(self.show_robot[name].value)
                    pelvis_position, pelvis_quaternion = pelvis_poses[name]
                    base.position = pelvis_position
                    base.wxyz = quaternion_xyzw_to_wxyz(pelvis_quaternion)
                    model = self.robot_models[name]
                    cfg = np.zeros(
                        len(model.get_actuated_joint_names()), dtype=np.float64
                    )
                    cfg[self.robot_cfg_indices[name]] = q
                    model.update_cfg(cfg)

            joint_index = G1_JOINT_NAMES.index(str(self.joint_name.value))
            self.current_q.value = float(q[joint_index])
            self.current_dq.value = float(dq[joint_index])
            self.current_tau.value = float(tau[joint_index])
            self.current_imu_rpy.value = tuple(float(value) for value in imu_rpy)

            position_error = None
            position_component_error = None
            orientation_error = None
            if self.glim is not None:
                (
                    position_error,
                    position_component_error,
                    orientation_error,
                ) = current_error(
                    self.glim, self.lio, sensor_timestamp
                )
                if self.current_position_error is not None:
                    self.current_position_error.value = position_error
                for handle, value in zip(
                    self.current_position_component_errors,
                    position_component_error,
                ):
                    handle.value = float(value)
                if self.current_orientation_error is not None:
                    self.current_orientation_error.value = orientation_error
            self._update_bag_augmented_map(elapsed)
            self._update_cloud_world(elapsed)

            now = time.monotonic()
            if not self.args.no_plots and (
                force_plots or now - self.last_plot_update >= 0.10
            ):
                if (
                    self.error_figure is not None
                    and self.error_plot_handle is not None
                    and position_error is not None
                    and position_component_error is not None
                ):
                    update_error_figure(
                        self.error_figure,
                        elapsed,
                        position_error,
                        position_component_error,
                    )
                    self.error_plot_handle.figure = self.error_figure
                for trajectory in self.trajectories:
                    _update_pose_figure(
                        self.pose_figures[trajectory.name],
                        trajectory,
                        self.common_start,
                        elapsed,
                    )
                    self.pose_plot_handles[trajectory.name].figure = (
                        self.pose_figures[trajectory.name]
                    )
                if self.joint_figure is not None and self.joint_plot_handle is not None:
                    _update_joint_figure(
                        self.joint_figure,
                        self.bag,
                        joint_index,
                        elapsed,
                        self.common_start,
                        self.clock_topic,
                    )
                    self.joint_plot_handle.figure = self.joint_figure
                self.last_plot_update = now

    def run_forever(self) -> None:
        print(f"Open http://localhost:{self.args.port} in a browser.", flush=True)
        speeds = {"0.25×": 0.25, "0.5×": 0.5, "1×": 1.0, "2×": 2.0}
        try:
            while True:
                time.sleep(0.02)
                self._update_live_augmented_map()
                with self.lock:
                    now = time.monotonic()
                    if not self.playing:
                        self.last_wall_time = now
                        continue
                    dt = now - self.last_wall_time
                    self.last_wall_time = now
                    value = float(self.time_slider.value) + dt * speeds[
                        str(self.speed_control.value)
                    ]
                    if value >= self.max_elapsed:
                        value = self.max_elapsed
                        self.playing = False
                    self.time_slider.value = value
                    self.update_state(value)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            if self.live_aug_receiver is not None:
                self.live_aug_receiver.close()


def print_dry_run_summary(
    bag: BagData,
    trajectories: list[Trajectory],
    lio_topic: str,
    glim_source: str,
    spatial_alignment: np.ndarray,
    map_points: np.ndarray,
    evaluation_bag: Path,
) -> None:
    lio = trajectories[-1]
    glim = trajectories[0] if len(trajectories) == 2 else None
    common_start = max(item.times[0] for item in trajectories)
    common_end = min(item.times[-1] for item in trajectories)
    print("\nValidation summary")
    print(f"  evaluated LIO bag: {evaluation_bag}")
    print(f"  replay context bag: {bag.path}")
    print(f"  odometry topics loaded: {sorted(bag.odometry)}")
    print(f"  selected LIO topic: {lio_topic}")
    print(f"  LIO filtering: {lio.filter_note}")
    print(f"  mode: {'GLIM vs Super LIO' if glim is not None else 'Super LIO only'}")
    if glim is not None:
        print(f"  GLIM source: {glim_source}")
    print(f"  lowstate samples: {len(bag.lowstate_times):,}")
    print(f"  TF pairs: {sorted(bag.transforms)}")
    print(
        "  augmented_map frames: "
        f"{len(bag.pointclouds['/augmented_map'].times) if '/augmented_map' in bag.pointclouds else 0}"
    )
    print(
        "  cloud_world frames: "
        f"{len(bag.pointclouds['/lio/cloud_world'].times) if '/lio/cloud_world' in bag.pointclouds else 0}"
    )
    print(f"  static map points: {len(map_points):,}")
    print(
        f"  common sensor interval: {common_start:.9f}..{common_end:.9f} "
        f"({common_end - common_start:.3f} s)"
    )
    if glim is not None:
        (
            elapsed,
            position_error,
            position_component_error,
            orientation_error,
        ) = compute_error_series(glim, lio, common_start, common_end)
        print(f"  graph samples (GLIM timestamps): {len(elapsed):,}")
        print(
            f"  position error mean/RMSE/max: {position_error.mean():.4f}/"
            f"{np.sqrt(np.mean(position_error**2)):.4f}/"
            f"{position_error.max():.4f} m"
        )
        print(
            f"  orientation error mean/RMSE/max: {orientation_error.mean():.3f}/"
            f"{np.sqrt(np.mean(orientation_error**2)):.3f}/"
            f"{orientation_error.max():.3f} deg"
        )
        component_mean = position_component_error.mean(axis=0)
        component_rmse = np.sqrt(np.mean(position_component_error**2, axis=0))
        print(
            "  signed xyz error mean: "
            f"{component_mean[0]:+.4f}/{component_mean[1]:+.4f}/"
            f"{component_mean[2]:+.4f} m (Super LIO - GLIM)"
        )
        print(
            "  xyz error RMSE: "
            f"{component_rmse[0]:.4f}/{component_rmse[1]:.4f}/"
            f"{component_rmse[2]:.4f} m"
        )
        alignment_end = getattr(
            glim, "post_recovery_alignment_end_elapsed", None
        )
        if alignment_end is not None:
            post_mask = elapsed > float(alignment_end)
            post_position = position_error[post_mask]
            post_orientation = orientation_error[post_mask]
            post_component = position_component_error[post_mask]
            print(
                f"  post-alignment-window samples: {len(post_position):,} "
                f"(elapsed > {float(alignment_end):.3f} s)"
            )
            print(
                "  post-window position mean/RMSE/max: "
                f"{post_position.mean():.4f}/"
                f"{np.sqrt(np.mean(post_position**2)):.4f}/"
                f"{post_position.max():.4f} m"
            )
            print(
                "  post-window signed xyz mean: "
                f"{post_component[:, 0].mean():+.4f}/"
                f"{post_component[:, 1].mean():+.4f}/"
                f"{post_component[:, 2].mean():+.4f} m"
            )
            print(
                "  post-window orientation mean/max: "
                f"{post_orientation.mean():.3f}/"
                f"{post_orientation.max():.3f} deg"
            )
        print(f"  spatial calibration: {glim.spatial_alignment_note}")
        print("  fixed T_MAP_GLIM transform:")
        print(
            np.array2string(spatial_alignment, precision=6, suppress_small=True)
        )
    else:
        print(f"  /lio/odom frames: {len(lio.times):,}")
        print("  GLIM alignment/error evaluation: skipped")


def main() -> None:
    args = parse_args()
    loaded = load_inputs(args)
    if args.dry_run:
        print_dry_run_summary(
            loaded[0],
            loaded[1],
            loaded[2],
            loaded[3],
            loaded[4],
            loaded[5],
            args.bag,
        )
        return
    app = DualReplayApp(args, *loaded)
    app.run_forever()


if __name__ == "__main__":
    main()

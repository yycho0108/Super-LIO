#!/usr/bin/env python3
"""Interactive Viser replay for G1 localization-stack evaluation."""

from __future__ import annotations

import argparse
from functools import partial
import os
from pathlib import Path
import threading
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

from replay_data import (
    apply_spatial_transform,
    BagData,
    G1_JOINT_NAMES,
    Trajectory,
    TrajectorySpec,
    align_trajectory_time,
    load_bag,
    load_odometry_only,
    load_pcd,
    load_trajectory,
    pointcloud2_xyz,
    pose_matrix,
    quaternion_xyzw_to_wxyz,
    remove_trajectory_pose_matches,
    robot_pelvis_pose,
)


PROJECT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_DIR / "assets"
DEFAULT_DATA_ROOT = Path(
    os.environ.get("LOCALIZATION_REPLAY_DATA_ROOT", Path.home())
).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay three localization trajectories, G1 lowstate, and a PCD map in Viser."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Base directory for default input paths. Default: $LOCALIZATION_REPLAY_DATA_ROOT or ~.",
    )
    parser.add_argument("--glim-imu", type=Path, default=None)
    parser.add_argument("--super-lio", type=Path, default=None)
    parser.add_argument(
        "--glim-corrected", type=Path, default=None
    )
    parser.add_argument("--bag-2026-08-02", type=Path, default=None)
    parser.add_argument("--bag-2026-08-03", type=Path, default=None)
    parser.add_argument(
        "--super-eval-bag",
        type=Path,
        default=None,
        help="Bag that recorded the Super-LIO localization run.",
    )
    parser.add_argument("--map", type=Path, default=ASSETS_DIR / "dump7_xyzi_clean.pcd")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=ASSETS_DIR / "g1_description/g1_29dof_rev_1_0.urdf",
    )
    parser.add_argument(
        "--super-time-offset",
        type=float,
        default=None,
        help="Override Super LIO raw-to-sensor clock offset in seconds. Default: use the evaluation bag.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-map-points", type=int, default=1_000_000)
    parser.add_argument("--point-size", type=float, default=0.025)
    parser.add_argument("--aug-map-point-size", type=float, default=0.02)
    parser.add_argument("--max-aug-map-points", type=int, default=200_000)
    parser.add_argument(
        "--live-aug-map-topic",
        default="/augmented_map",
        help="Live ROS PointCloud2 topic to overlay. Default: /augmented_map.",
    )
    parser.add_argument("--no-live-aug-map", action="store_true")
    parser.add_argument(
        "--no-glim-spatial-align",
        action="store_true",
        help="Keep both GLIM trajectories in their raw coordinate frame.",
    )
    parser.add_argument("--no-robot", action="store_true", help="Skip loading the URDF meshes.")
    parser.add_argument("--no-plots", action="store_true", help="Skip Plotly panels.")
    parser.add_argument("--dry-run", action="store_true", help="Load and validate all data without starting Viser.")
    args = parser.parse_args()
    data_root = args.data_root.expanduser()
    defaults = {
        "glim_imu": data_root / "Downloads/glim_dump8/traj_imu.txt",
        "super_lio": data_root / "super_lio_trajectory.txt",
        "glim_corrected": data_root / "glim_ros_odom_corrected_trajectory.txt",
        "bag_2026_08_02": data_root / "Downloads/rosbag2_2026_08_02-12_41_25",
        "bag_2026_08_03": data_root / "Downloads/rosbag2_2026_08_03-15_13_43",
        "super_eval_bag": data_root / "rosbag2_2026_08_03-22_09_23",
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def trajectory_specs(args: argparse.Namespace) -> list[TrajectorySpec]:
    return [
        TrajectorySpec(
            name="GLIM IMU (08-03)",
            path=args.glim_imu,
            bag_key="08-03",
            quat_order="xyzw",
            color=(239, 83, 80),
            match_topic="/glim_ros/odom",
            time_mode="raw_sensor",
        ),
        TrajectorySpec(
            name="Super LIO (08-02)",
            path=args.super_lio,
            bag_key="08-02",
            quat_order="wxyz",
            color=(67, 190, 113),
            match_topic="/lio/odom",
            time_mode="record_to_sensor",
        ),
        TrajectorySpec(
            name="GLIM corrected (08-03)",
            path=args.glim_corrected,
            bag_key="08-03",
            quat_order="wxyz",
            color=(65, 135, 245),
            match_topic="/glim_ros/odom_corrected",
            time_mode="record_to_sensor",
        ),
    ]


def load_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, BagData], list[Trajectory], np.ndarray, np.ndarray, np.ndarray]:
    specs = trajectory_specs(args)
    print("[1/5] Loading rosbag 08-02 (/lowstate, TF, odometry)...", flush=True)
    bag_02 = load_bag(
        "08-02",
        args.bag_2026_08_02,
        odometry_topics=("/lio/odom", "/odometry/odom"),
        pointcloud_topics=("/augmented_map",),
    )
    print("[2/5] Loading rosbag 08-03 (/lowstate, TF, odometry)...", flush=True)
    bag_03 = load_bag(
        "08-03",
        args.bag_2026_08_03,
        odometry_topics=(
            "/glim_ros/odom",
            "/glim_ros/odom_corrected",
            "/odometry/odom",
        ),
        pointcloud_topics=("/augmented_map",),
    )
    bags = {"08-02": bag_02, "08-03": bag_03}
    print("[3/5] Loading Super-LIO evaluation odometry clock...", flush=True)
    super_eval_odometry = load_odometry_only(
        args.super_eval_bag, ("/lio/odom",)
    )["/lio/odom"]

    print("[4/5] Loading, filtering, and aligning trajectory text files...", flush=True)
    trajectories = [load_trajectory(spec) for spec in specs]
    super_lio = next(
        item for item in trajectories if item.name.startswith("Super LIO")
    )
    removed = remove_trajectory_pose_matches(
        super_lio, bag_02.odometry["/lio/odom"]
    )
    if removed == 0:
        raise RuntimeError(
            "Super-LIO trajectory contains no replayed source poses; "
            "input assumptions may have changed"
        )
    for trajectory in trajectories:
        forced = args.super_time_offset if trajectory.name.startswith("Super LIO") else None
        clock_override = (
            super_eval_odometry
            if trajectory.name.startswith("Super LIO")
            else None
        )
        align_trajectory_time(
            trajectory,
            bags[trajectory.spec.bag_key],
            forced_offset=forced,
            odometry_override=clock_override,
        )
        print(
            f"  {trajectory.name}: {len(trajectory.times)} poses, sensor time "
            f"{trajectory.times[0]:.6f}..{trajectory.times[-1]:.6f}, "
            f"{trajectory.alignment_note}; {trajectory.filter_note}",
            flush=True,
        )

    common_start = max(item.times[0] for item in trajectories)
    common_end = min(item.times[-1] for item in trajectories)
    if common_end <= common_start:
        raise RuntimeError("The three trajectories have no common sensor timestamp interval")
    print(
        f"  Exact common sensor-time interval: {common_start:.6f}..{common_end:.6f} "
        f"({common_end - common_start:.3f} s)",
        flush=True,
    )

    spatial_alignment = np.eye(4, dtype=np.float64)
    if not args.no_glim_spatial_align:
        map_odometry = bag_02.odometry["/odometry/odom"]
        reference_odometry = bag_03.odometry["/odometry/odom"]
        calibration_timestamp = float(reference_odometry.header_times[0])
        map_position, map_quaternion = map_odometry.sample_header_time(
            calibration_timestamp
        )
        reference_position, reference_quaternion = (
            reference_odometry.sample_header_time(calibration_timestamp)
        )
        spatial_alignment = pose_matrix(
            map_position, map_quaternion
        ) @ np.linalg.inv(pose_matrix(reference_position, reference_quaternion))
        alignment_note = (
            "independent one-pose map calibration from /odometry/odom at "
            f"sensor timestamp {calibration_timestamp:.6f}; fixed for every pose"
        )
        for trajectory in trajectories:
            if trajectory.name.startswith("GLIM"):
                apply_spatial_transform(trajectory, spatial_alignment, alignment_note)
        print(
            "  Fixed GLIM -> map alignment from independent /odometry/odom "
            "calibration (shared by both GLIM paths):",
            flush=True,
        )
        print(np.array2string(spatial_alignment, precision=6, suppress_small=True), flush=True)

    print("[5/5] Loading PCD map...", flush=True)
    points, colors, _ = load_pcd(args.map, max_points=args.max_map_points)
    print(f"  Map: {len(points):,} points", flush=True)
    return bags, trajectories, points, colors, spatial_alignment


def _pose_figure(trajectory: Trajectory, common_start: float) -> go.Figure:
    elapsed = trajectory.times - common_start
    euler = Rotation.from_quat(trajectory.quaternions_xyzw).as_euler("xyz", degrees=True)
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Position [m]", "Orientation RPY [deg]"),
    )
    axis_colors = ("#EF5350", "#43A047", "#4285F4")
    for axis, color, index in zip(("x", "y", "z"), axis_colors, range(3)):
        figure.add_trace(
            go.Scattergl(x=elapsed, y=trajectory.positions[:, index], name=f"p{axis}", line={"color": color, "width": 1.4}),
            row=1,
            col=1,
        )
    for axis, color, index in zip(("roll", "pitch", "yaw"), axis_colors, range(3)):
        figure.add_trace(
            go.Scattergl(x=elapsed, y=euler[:, index], name=axis, line={"color": color, "width": 1.2, "dash": "dot"}),
            row=2,
            col=1,
        )
    for axis, color, index in zip(("x", "y", "z"), axis_colors, range(3)):
        figure.add_trace(
            go.Scattergl(
                x=[float(elapsed[0])],
                y=[float(trajectory.positions[0, index])],
                name=f"current p{axis}",
                mode="markers",
                marker={"color": color, "size": 9, "line": {"color": "#F5C542", "width": 2}},
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    for axis, color, index in zip(("roll", "pitch", "yaw"), axis_colors, range(3)):
        figure.add_trace(
            go.Scattergl(
                x=[float(elapsed[0])],
                y=[float(euler[0, index])],
                name=f"current {axis}",
                mode="markers",
                marker={"color": color, "size": 9, "line": {"color": "#F5C542", "width": 2}},
                showlegend=False,
            ),
            row=2,
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
        line={"color": "#F5C542", "width": 2},
    )
    figure.update_layout(
        title={"text": trajectory.name, "font": {"size": 15}},
        height=430,
        margin={"l": 42, "r": 12, "t": 45, "b": 35},
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        hovermode="x unified",
        template="plotly_dark",
    )
    figure.update_xaxes(title_text="bag elapsed time [s]", row=2, col=1)
    return figure


def _joint_figure(
    bag: BagData,
    joint_index: int,
    cursor: float,
    common_start: float,
    clock_topic: str,
) -> go.Figure:
    sensor_times = bag.sensor_time_from_record_time(bag.lowstate_times, clock_topic)
    elapsed = sensor_times - common_start
    stride = max(1, len(elapsed) // 2500)
    selected = slice(None, None, stride)
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("q [rad]", "dq [rad/s]", "estimated torque [Nm]"),
    )
    for row, values, color, label in (
        (1, bag.q[:, joint_index], "#66BB6A", "q"),
        (2, bag.dq[:, joint_index], "#42A5F5", "dq"),
        (3, bag.tau[:, joint_index], "#EF5350", "tau_est"),
    ):
        figure.add_trace(
            go.Scattergl(
                x=elapsed[selected],
                y=values[selected],
                name=label,
                line={"color": color, "width": 1.1},
            ),
            row=row,
            col=1,
        )
    current_sensor_time = common_start + cursor
    current_record_time = float(
        bag.record_time_from_sensor_time(current_sensor_time, clock_topic)
    )
    current_index = bag.lowstate_index(current_record_time)
    for row, values, color, label in (
        (1, bag.q[:, joint_index], "#66BB6A", "current q"),
        (2, bag.dq[:, joint_index], "#42A5F5", "current dq"),
        (3, bag.tau[:, joint_index], "#EF5350", "current tau"),
    ):
        figure.add_trace(
            go.Scattergl(
                x=[cursor],
                y=[float(values[current_index])],
                name=label,
                mode="markers",
                marker={"color": color, "size": 10, "line": {"color": "#F5C542", "width": 2}},
                showlegend=False,
            ),
            row=row,
            col=1,
        )
    figure.add_shape(
        type="line",
        x0=cursor,
        x1=cursor,
        y0=0.0,
        y1=1.0,
        xref="x",
        yref="paper",
        line={"color": "#F5C542", "width": 2},
    )
    figure.update_layout(
        title={"text": f"{bag.key} · {G1_JOINT_NAMES[joint_index]}", "font": {"size": 15}},
        height=510,
        margin={"l": 42, "r": 12, "t": 45, "b": 35},
        showlegend=False,
        hovermode="x unified",
        template="plotly_dark",
    )
    figure.update_xaxes(title_text="bag elapsed time [s]", row=3, col=1)
    return figure


def _set_figure_cursor(figure: go.Figure, elapsed: float) -> None:
    if figure.layout.shapes:
        figure.layout.shapes[0].x0 = elapsed
        figure.layout.shapes[0].x1 = elapsed


def _update_pose_figure(
    figure: go.Figure,
    trajectory: Trajectory,
    common_start: float,
    elapsed: float,
) -> None:
    _set_figure_cursor(figure, elapsed)
    position, quaternion, _ = trajectory.sample(common_start + elapsed)
    rpy = Rotation.from_quat(quaternion).as_euler("xyz", degrees=True)
    for index in range(3):
        figure.data[6 + index].x = (elapsed,)
        figure.data[6 + index].y = (float(position[index]),)
        figure.data[9 + index].x = (elapsed,)
        figure.data[9 + index].y = (float(rpy[index]),)


def _update_joint_figure(
    figure: go.Figure,
    bag: BagData,
    joint_index: int,
    elapsed: float,
    common_start: float,
    clock_topic: str,
) -> None:
    _set_figure_cursor(figure, elapsed)
    record_time = float(
        bag.record_time_from_sensor_time(common_start + elapsed, clock_topic)
    )
    index = bag.lowstate_index(record_time)
    for trace_index, values in zip(
        range(3, 6),
        (bag.q[:, joint_index], bag.dq[:, joint_index], bag.tau[:, joint_index]),
    ):
        figure.data[trace_index].x = (elapsed,)
        figure.data[trace_index].y = (float(values[index]),)


def _sample_trajectory_many(
    trajectory: Trajectory, timestamps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    samples = [trajectory.sample(float(timestamp))[:2] for timestamp in timestamps]
    return (
        np.stack([sample[0] for sample in samples]),
        np.stack([sample[1] for sample in samples]),
    )


def _compute_error_series(
    trajectories: list[Trajectory],
    common_start: float,
    common_end: float,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    sample_count = max(2, int(np.ceil((common_end - common_start) * 30.0)) + 1)
    sensor_times = np.linspace(common_start, common_end, sample_count)
    elapsed = sensor_times - common_start
    sampled = {
        trajectory.name: _sample_trajectory_many(trajectory, sensor_times)
        for trajectory in trajectories
    }
    super_name = next(name for name in sampled if name.startswith("Super LIO"))
    glim_name = next(name for name in sampled if name.startswith("GLIM IMU"))
    corrected_name = next(name for name in sampled if name.startswith("GLIM corrected"))
    pairs = {
        "GLIM IMU ↔ Super LIO": (glim_name, super_name),
        "GLIM corrected ↔ Super LIO": (corrected_name, super_name),
        "GLIM corrected ↔ GLIM IMU": (corrected_name, glim_name),
    }
    errors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, (first_name, second_name) in pairs.items():
        first_position, first_quaternion = sampled[first_name]
        second_position, second_quaternion = sampled[second_name]
        position_error = np.linalg.norm(first_position - second_position, axis=1)
        relative_rotation = (
            Rotation.from_quat(second_quaternion).inv()
            * Rotation.from_quat(first_quaternion)
        )
        orientation_error = np.rad2deg(relative_rotation.magnitude())
        errors[label] = (position_error, orientation_error)
    return elapsed, errors


def _error_figure(
    elapsed: np.ndarray,
    errors: dict[str, tuple[np.ndarray, np.ndarray]],
) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Same-timestamp position error [m]", "Same-timestamp orientation error [deg]"),
    )
    colors = ("#EF5350", "#4285F4", "#AB47BC")
    dashes = ("solid", "solid", "dot")
    for (label, (position_error, _)), color, dash in zip(errors.items(), colors, dashes):
        figure.add_trace(
            go.Scattergl(
                x=elapsed,
                y=position_error,
                name=label,
                line={"color": color, "width": 1.8, "dash": dash},
            ),
            row=1,
            col=1,
        )
    for (label, (_, orientation_error)), color, dash in zip(errors.items(), colors, dashes):
        figure.add_trace(
            go.Scattergl(
                x=elapsed,
                y=orientation_error,
                name=label + " orientation",
                line={"color": color, "width": 1.6, "dash": dash},
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    for row, metric_index in ((1, 0), (2, 1)):
        for (label, values), color in zip(errors.items(), colors):
            figure.add_trace(
                go.Scattergl(
                    x=[0.0],
                    y=[float(values[metric_index][0])],
                    name="current " + label,
                    mode="markers",
                    marker={
                        "color": color,
                        "size": 11,
                        "line": {"color": "#F5C542", "width": 2},
                    },
                    showlegend=False,
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
        editable=True,
        line={"color": "#F5C542", "width": 5},
    )
    figure.update_layout(
        title={
            "text": "Exact common-timestamp errors · click plot or drag yellow cursor",
            "font": {"size": 15},
        },
        height=500,
        margin={"l": 48, "r": 12, "t": 48, "b": 38},
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        hovermode="x unified",
        dragmode="pan",
        meta={"g1_error_timeline": True},
        template="plotly_dark",
    )
    figure.update_xaxes(title_text="exact common sensor time from overlap start [s]", row=2, col=1)
    return figure


def _update_error_figure(
    figure: go.Figure,
    error_elapsed: np.ndarray,
    errors: dict[str, tuple[np.ndarray, np.ndarray]],
    elapsed: float,
) -> dict[str, tuple[float, float]]:
    _set_figure_cursor(figure, elapsed)
    current: dict[str, tuple[float, float]] = {}
    for pair_index, (label, (position_error, orientation_error)) in enumerate(errors.items()):
        position_value = float(np.interp(elapsed, error_elapsed, position_error))
        orientation_value = float(np.interp(elapsed, error_elapsed, orientation_error))
        figure.data[6 + pair_index].x = (elapsed,)
        figure.data[6 + pair_index].y = (position_value,)
        figure.data[9 + pair_index].x = (elapsed,)
        figure.data[9 + pair_index].y = (orientation_value,)
        current[label] = (position_value, orientation_value)
    return current


class LiveAugmentedMapReceiver:
    """Background ROS 2 subscriber that hands the newest PointCloud2 to Viser."""

    def __init__(self, topic: str, max_points: int) -> None:
        self.topic = topic
        self.max_points = max_points
        self.lock = threading.Lock()
        self.version = 0
        self.latest: tuple[np.ndarray, float, str] | None = None
        self.error: str | None = None
        self.node = None
        self._owns_rclpy = False
        try:
            import rclpy
            from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import PointCloud2

            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_rclpy = True
            self.node = rclpy.create_node("g1_localization_replay_augmented_map")
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.node.create_subscription(PointCloud2, topic, self._callback, qos)
            self.thread = threading.Thread(target=self._spin, daemon=True, name="live-aug-map")
            self.thread.start()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def _callback(self, message) -> None:
        try:
            points = pointcloud2_xyz(message, max_points=self.max_points)
            stamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
            with self.lock:
                self.latest = (points, stamp, str(message.header.frame_id))
                self.version += 1
        except Exception as exc:
            self.error = f"PointCloud2 conversion failed: {type(exc).__name__}: {exc}"

    def _spin(self) -> None:
        try:
            import rclpy

            if self.node is not None:
                rclpy.spin(self.node)
        except Exception as exc:
            self.error = f"ROS spin stopped: {type(exc).__name__}: {exc}"

    def snapshot(self, last_version: int) -> tuple[int, np.ndarray, float, str] | None:
        with self.lock:
            if self.version == last_version or self.latest is None:
                return None
            points, stamp, frame_id = self.latest
            return self.version, points, stamp, frame_id

    def close(self) -> None:
        try:
            import rclpy

            if self.node is not None:
                self.node.destroy_node()
            if self._owns_rclpy and rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


class ReplayApp:
    def __init__(
        self,
        args: argparse.Namespace,
        bags: dict[str, BagData],
        trajectories: list[Trajectory],
        map_points: np.ndarray,
        map_colors: np.ndarray,
        spatial_alignment: np.ndarray,
    ) -> None:
        import viser

        self.args = args
        self.bags = bags
        self.trajectories = trajectories
        self.spatial_alignment = spatial_alignment
        self.clock_topics = {"08-02": "/lio/odom", "08-03": "/glim_ros/odom"}
        self.common_start = max(trajectory.times[0] for trajectory in trajectories)
        self.common_end = min(trajectory.times[-1] for trajectory in trajectories)
        if self.common_end <= self.common_start:
            raise RuntimeError("The three trajectories have no exact common timestamp interval")
        self.server = viser.ViserServer(host=args.host, port=args.port, label="G1 localization replay")
        self.server.scene.set_up_direction("+z")
        self.lock = threading.RLock()
        self.playing = False
        self.last_wall_time = time.monotonic()
        self.last_plot_update = 0.0

        self.max_elapsed = self.common_end - self.common_start
        self.error_elapsed, self.errors = _compute_error_series(
            trajectories,
            self.common_start,
            self.common_end,
        )
        self.map_handle = self.server.scene.add_point_cloud(
            "/map/dump7_xyzi_clean",
            points=map_points,
            colors=map_colors,
            point_size=args.point_size,
            point_shape="circle",
            precision="float32",
        )
        self.server.scene.add_grid(
            "/map/ground_grid",
            width=20.0,
            height=20.0,
            plane="xy",
            cell_size=0.5,
            cell_thickness=0.5,
            section_size=2.0,
            section_thickness=1.0,
            position=(0.0, 0.0, -0.5),
        )

        self.pose_frames = {}
        self.pose_markers = {}
        self.trajectory_lines = {}
        for index, trajectory in enumerate(trajectories):
            slug = f"trajectory_{index}"
            color = trajectory.spec.color
            points = trajectory.positions.astype(np.float32)
            line_points = np.stack((points[:-1], points[1:]), axis=1)
            self.trajectory_lines[trajectory.name] = self.server.scene.add_line_segments(
                f"/trajectories/{slug}/path",
                points=line_points,
                colors=color,
                line_width=3.0,
            )
            self.pose_frames[trajectory.name] = self.server.scene.add_frame(
                f"/trajectories/{slug}/imu_pose",
                axes_length=0.22,
                axes_radius=0.008,
            )
            self.pose_markers[trajectory.name] = self.server.scene.add_icosphere(
                f"/trajectories/{slug}/current_point",
                radius=0.075,
                color=color,
            )
            self.server.scene.add_label(
                f"/trajectories/{slug}/imu_pose/label",
                text=trajectory.name,
                position=(0.0, 0.0, 0.18),
            )

        self.robot_bases = {}
        self.robot_models = {}
        self.robot_cfg_indices: dict[str, np.ndarray] = {}
        if not args.no_robot:
            self._load_robots(args.urdf)

        self.bag_aug_handles = {key: None for key in self.bags}
        self.bag_aug_last_index: dict[str, int | None] = {key: None for key in self.bags}
        self.live_aug_handle = None
        self.live_aug_version = 0
        self.live_aug_receiver = None
        if not args.no_live_aug_map:
            self.live_aug_receiver = LiveAugmentedMapReceiver(
                args.live_aug_map_topic,
                args.max_aug_map_points,
            )

        self._build_gui()
        self._register_client_camera(map_points)
        self.update_state(0.0, force_plots=True)

    def _replace_augmented_map(
        self,
        handle,
        name: str,
        points: np.ndarray,
        color: tuple[int, int, int],
        visible: bool,
    ):
        if handle is not None:
            # Keep the existing scene node so browser-side point-size edits are
            # not discarded whenever playback selects another bag frame.
            handle.points = np.asarray(points, dtype=np.float32)
            handle.colors = np.asarray(color, dtype=np.uint8)
            handle.visible = visible
            return handle
        point_size = float(
            self.aug_map_point_size.value
            if hasattr(self, "aug_map_point_size")
            else self.args.aug_map_point_size
        )
        return self.server.scene.add_point_cloud(
            name,
            points=points,
            colors=color,
            point_size=point_size,
            point_shape="circle",
            precision="float32",
            visible=visible,
        )

    def _update_bag_augmented_maps(self, elapsed: float) -> None:
        colors = {"08-02": (255, 170, 55), "08-03": (65, 220, 235)}
        for key, bag in self.bags.items():
            control = self.show_bag_aug_map[key]
            handle = self.bag_aug_handles[key]
            if not bool(control.value):
                if handle is not None:
                    handle.visible = False
                continue
            series = bag.pointclouds.get("/augmented_map")
            if series is None:
                self.bag_aug_status[key].value = "topic not present"
                continue
            sensor_timestamp = self.common_start + elapsed
            record_timestamp = float(
                bag.record_time_from_sensor_time(sensor_timestamp, self.clock_topics[key])
            )
            index = series.frame_index_at(record_timestamp)
            if index is None:
                if handle is not None:
                    handle.visible = False
                self.bag_aug_status[key].value = "waiting for first bag frame"
                continue
            if index != self.bag_aug_last_index[key] or handle is None:
                frame = series.load_frame(index, max_points=self.args.max_aug_map_points)
                handle = self._replace_augmented_map(
                    handle,
                    f"/augmented_map/bag_{key.replace('-', '_')}",
                    frame.points,
                    colors[key],
                    True,
                )
                self.bag_aug_handles[key] = handle
                self.bag_aug_last_index[key] = index
                self.bag_aug_status[key].value = (
                    f"frame {index + 1}/{len(series.times)}, {len(frame.points):,} points, "
                    f"frame_id={frame.frame_id or '<empty>'}"
                )
            else:
                handle.visible = True

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
        self.live_aug_handle = self._replace_augmented_map(
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

    def _load_robots(self, urdf_path: Path) -> None:
        import yourdfpy
        from viser.extras import ViserUrdf

        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        print("Loading G1 URDF meshes (three colored localization hypotheses)...", flush=True)
        urdf = yourdfpy.URDF.load(
            urdf_path,
            build_scene_graph=True,
            build_collision_scene_graph=False,
            load_meshes=True,
            load_collision_meshes=False,
            filename_handler=partial(yourdfpy.filename_handler_magic, dir=urdf_path.parent),
        )
        colors = (
            (0.94, 0.33, 0.31, 0.72),
            (0.26, 0.75, 0.44, 0.72),
            (0.25, 0.53, 0.96, 0.72),
        )
        for index, (trajectory, color) in enumerate(zip(self.trajectories, colors)):
            root = f"/robots/robot_{index}"
            self.robot_bases[trajectory.name] = self.server.scene.add_frame(root, show_axes=False)
            model = ViserUrdf(
                self.server,
                urdf,
                root_node_name=root,
                mesh_color_override=color,
            )
            self.robot_models[trajectory.name] = model
            urdf_joint_names = model.get_actuated_joint_names()
            name_to_index = {name: i for i, name in enumerate(urdf_joint_names)}
            self.robot_cfg_indices[trajectory.name] = np.array(
                [name_to_index[name] for name in G1_JOINT_NAMES], dtype=np.int64
            )

    def _build_gui(self) -> None:
        gui = self.server.gui
        gui.add_markdown(
            "# G1 localization replay\n"
            "세 결과는 원본 센서의 **동일한 header timestamp**로 보간·비교됩니다. "
            "Super LIO는 재생 원본 pose를 제외한 localization 출력만 사용합니다. "
            "색상: GLIM IMU=빨강, Super LIO=초록, corrected GLIM=파랑."
        )
        with gui.add_folder("Fixed GLIM spatial alignment", expand_by_default=False):
            calibration_timestamp = float(
                self.bags["08-03"].odometry["/odometry/odom"].header_times[0]
            )
            gui.add_markdown(
                "평가 trajectory를 사용하지 않고 두 bag의 `/odometry/odom` pose 한 쌍으로 "
                "고정 변환을 구합니다. 두 GLIM 경로에 모든 시각에서 이 변환 하나만 적용합니다.\n\n"
                "`T_map_glim = T_map_odom(08-02,t_cal) · "
                "inv(T_glim_odom(08-03,t_cal))`, "
                f"`t_cal={calibration_timestamp:.9f}`\n\n"
                "```text\n"
                + np.array2string(self.spatial_alignment, precision=6, suppress_small=True)
                + "\n```"
            )
        with gui.add_folder("Playback", expand_by_default=True):
            self.time_slider = gui.add_slider(
                "exact common timestamp offset [s]",
                min=0.0,
                max=self.max_elapsed,
                step=0.01,
                initial_value=0.0,
                marks=((0.0, "0"), (round(self.max_elapsed, 2), f"{self.max_elapsed:.1f}s")),
            )
            self.play_button = gui.add_button("Play / Pause", icon="player-play")
            self.speed_control = gui.add_button_group("speed", options=("0.25×", "0.5×", "1×", "2×"))
            self.speed_control.value = "1×"
            self.elapsed_number = gui.add_number("current elapsed [s]", 0.0, disabled=True)
            self.sensor_timestamp = gui.add_number(
                "exact sensor timestamp", self.common_start, disabled=True
            )
            self.timestamp_02 = gui.add_number(
                "08-02 bag record time",
                float(
                    self.bags["08-02"].record_time_from_sensor_time(
                        self.common_start, self.clock_topics["08-02"]
                    )
                ),
                disabled=True,
            )
            self.timestamp_03 = gui.add_number(
                "08-03 bag record time",
                float(
                    self.bags["08-03"].record_time_from_sensor_time(
                        self.common_start, self.clock_topics["08-03"]
                    )
                ),
                disabled=True,
            )

        with gui.add_folder("Visibility", expand_by_default=False):
            gui.add_markdown(
                "trajectory 선과 pose frame은 **IMU 센서 위치**이므로 바닥보다 높게 보이는 것이 "
                "정상입니다. G1 mesh는 해당 시각의 TF로 pelvis 위치까지 변환해 표시합니다."
            )
            self.show_map = gui.add_checkbox("map point cloud", True)
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

        with gui.add_folder("Augmented maps", expand_by_default=True):
            gui.add_markdown(
                "bag cloud는 slider 시각 이전의 최신 `/augmented_map` 프레임입니다. "
                "토픽 자체가 Super/LIO `map` frame이므로 GLIM 정렬 변환은 적용하지 않습니다."
            )
            self.aug_map_point_size = gui.add_slider(
                "point size",
                min=0.005,
                max=0.08,
                step=0.005,
                initial_value=float(self.args.aug_map_point_size),
            )
            self.show_bag_aug_map = {}
            self.bag_aug_status = {}
            for key in ("08-02", "08-03"):
                available = "/augmented_map" in self.bags[key].pointclouds
                self.show_bag_aug_map[key] = gui.add_checkbox(
                    f"bag {key} /augmented_map",
                    available,
                    disabled=not available,
                )
                self.bag_aug_status[key] = gui.add_text(
                    f"bag {key} status",
                    "waiting for first bag frame" if available else "topic not present",
                    disabled=True,
                )
            live_available = (
                self.live_aug_receiver is not None and self.live_aug_receiver.error is None
            )
            self.show_live_aug_map = gui.add_checkbox(
                f"live {self.args.live_aug_map_topic}",
                live_available,
                disabled=not live_available,
            )
            initial_live_status = "waiting for ROS message"
            if self.live_aug_receiver is not None and self.live_aug_receiver.error is not None:
                initial_live_status = self.live_aug_receiver.error
            if self.args.no_live_aug_map:
                initial_live_status = "disabled by --no-live-aug-map"
            self.live_aug_status = gui.add_text(
                "live status",
                initial_live_status,
                disabled=True,
            )

        self.pose_gui = {}
        with gui.add_folder("Current trajectory values", expand_by_default=True):
            for trajectory in self.trajectories:
                with gui.add_folder(trajectory.name, expand_by_default=False):
                    position = gui.add_vector3("position xyz [m]", (0.0, 0.0, 0.0), disabled=True)
                    rpy = gui.add_vector3("orientation rpy [deg]", (0.0, 0.0, 0.0), disabled=True)
                    index = gui.add_number("text row", 0, disabled=True)
                    sample_time = gui.add_number("query sensor timestamp", 0.0, disabled=True)
                    nearest_dt = gui.add_number("nearest row dt [ms]", 0.0, disabled=True)
                    status = gui.add_text("status", "not started", disabled=True)
                    self.pose_gui[trajectory.name] = (
                        position,
                        rpy,
                        index,
                        sample_time,
                        nearest_dt,
                        status,
                    )

        with gui.add_folder("LowState joint inspector", expand_by_default=True):
            self.joint_bag = gui.add_dropdown("session", options=("08-02", "08-03"), initial_value="08-02")
            self.joint_name = gui.add_dropdown(
                "joint", options=G1_JOINT_NAMES, initial_value="left_hip_pitch_joint"
            )
            self.current_q = gui.add_number("q [rad]", 0.0, disabled=True)
            self.current_dq = gui.add_number("dq [rad/s]", 0.0, disabled=True)
            self.current_tau = gui.add_number("tau_est [Nm]", 0.0, disabled=True)
            self.current_imu_rpy = gui.add_vector3("lowstate IMU rpy [rad]", (0.0, 0.0, 0.0), disabled=True)

        self.pose_figures: dict[str, go.Figure] = {}
        self.pose_plot_handles = {}
        self.error_figure: go.Figure | None = None
        self.error_plot_handle = None
        self.joint_figure: go.Figure | None = None
        self.joint_plot_handle = None
        self.current_error_gui = {}
        with gui.add_folder("Exact same-timestamp error", expand_by_default=True):
            gui.add_markdown(
                "그래프를 클릭하면 해당 timestamp로 이동합니다. 굵은 노란 cursor를 잡아 "
                "좌우로 drag하면 로봇 세 개와 모든 current marker가 동시에 움직입니다."
            )
            for label in self.errors:
                with gui.add_folder(label, expand_by_default=False):
                    position_error = gui.add_number("position error [m]", 0.0, disabled=True)
                    orientation_error = gui.add_number(
                        "orientation error [deg]", 0.0, disabled=True
                    )
                    self.current_error_gui[label] = (position_error, orientation_error)
            if not self.args.no_plots:
                self.error_figure = _error_figure(self.error_elapsed, self.errors)
                self.error_plot_handle = gui.add_plotly(
                    self.error_figure,
                    aspect=1.6,
                    config={
                        "displaylogo": False,
                        "responsive": True,
                        "editable": True,
                        "edits": {"shapePosition": True},
                        "scrollZoom": True,
                    },
                )
        if not self.args.no_plots:
            with gui.add_folder("Trajectory plots", expand_by_default=False):
                for trajectory in self.trajectories:
                    figure = _pose_figure(trajectory, self.common_start)
                    self.pose_figures[trajectory.name] = figure
                    self.pose_plot_handles[trajectory.name] = gui.add_plotly(
                        figure, aspect=1.75, config={"displaylogo": False, "responsive": True}
                    )
            with gui.add_folder("Selected joint plot", expand_by_default=False):
                self.joint_figure = _joint_figure(
                    self.bags["08-02"],
                    0,
                    0.0,
                    self.common_start,
                    self.clock_topics["08-02"],
                )
                self.joint_plot_handle = gui.add_plotly(
                    self.joint_figure, aspect=1.5, config={"displaylogo": False, "responsive": True}
                )
            self._install_error_plot_drag_bridge()

        @self.time_slider.on_update
        def _(_) -> None:
            self.update_state(float(self.time_slider.value))

        @self.play_button.on_click
        def _(_) -> None:
            with self.lock:
                self.playing = not self.playing
                self.last_wall_time = time.monotonic()

        @self.show_map.on_update
        def _(_) -> None:
            self.map_handle.visible = bool(self.show_map.value)

        for trajectory_name, control in self.show_robot.items():
            @control.on_update
            def _(_, name=trajectory_name, checkbox=control) -> None:
                if name in self.robot_bases:
                    self.robot_bases[name].visible = bool(checkbox.value)

        @self.show_paths.on_update
        def _(_) -> None:
            for path in self.trajectory_lines.values():
                path.visible = bool(self.show_paths.value)

        @self.show_frames.on_update
        def _(_) -> None:
            for frame in self.pose_frames.values():
                frame.visible = bool(self.show_frames.value)
            for marker in self.pose_markers.values():
                marker.visible = bool(self.show_frames.value)

        for key, control in self.show_bag_aug_map.items():
            @control.on_update
            def _(_, bag_key=key) -> None:
                handle = self.bag_aug_handles[bag_key]
                if handle is not None:
                    handle.visible = bool(self.show_bag_aug_map[bag_key].value)
                if bool(self.show_bag_aug_map[bag_key].value):
                    self._update_bag_augmented_maps(float(self.time_slider.value))

        @self.show_live_aug_map.on_update
        def _(_) -> None:
            if self.live_aug_handle is not None:
                self.live_aug_handle.visible = bool(self.show_live_aug_map.value)

        @self.aug_map_point_size.on_update
        def _(_) -> None:
            point_size = float(self.aug_map_point_size.value)
            self.args.aug_map_point_size = point_size
            for handle in self.bag_aug_handles.values():
                if handle is not None:
                    handle.point_size = point_size
            if self.live_aug_handle is not None:
                self.live_aug_handle.point_size = point_size

        @self.joint_bag.on_update
        def _(_) -> None:
            self._rebuild_joint_plot(float(self.time_slider.value))
            self.update_state(float(self.time_slider.value), force_plots=True)

        @self.joint_name.on_update
        def _(_) -> None:
            self._rebuild_joint_plot(float(self.time_slider.value))
            self.update_state(float(self.time_slider.value), force_plots=True)

    def _register_client_camera(self, map_points: np.ndarray) -> None:
        low = np.percentile(map_points, 2.0, axis=0)
        high = np.percentile(map_points, 98.0, axis=0)
        center = (low + high) * 0.5
        radius = max(float(np.linalg.norm(high - low)), 4.0)

        @self.server.on_client_connect
        def _(client) -> None:
            client.camera.up_direction = (0.0, 0.0, 1.0)
            client.camera.look_at = center
            client.camera.position = center + np.array((0.55 * radius, -0.75 * radius, 0.55 * radius))

    def _install_error_plot_drag_bridge(self) -> None:
        """Bridge Plotly click/shape-drag events to the synchronized Viser slider."""
        from viser import _messages as viser_messages

        slider_uuid = self.time_slider._impl.uuid
        script = f"""
(() => {{
  if (window.__g1ErrorTimelineBridge) {{
    clearInterval(window.__g1ErrorTimelineBridge);
  }}
  const sliderId = {slider_uuid!r};
  const maxTime = {self.max_elapsed!r};
  let lastSent = 0;
  function setTimeline(value) {{
    value = Math.max(0, Math.min(maxTime, Number(value)));
    if (!Number.isFinite(value)) return;
    const now = performance.now();
    if (now - lastSent < 20) return;
    lastSent = now;
    const slider = document.getElementById(sliderId);
    if (!slider || !slider.parentElement) return;
    const input = slider.parentElement.querySelector('input[type="text"]');
    if (!input) return;
    const oldValue = input.value;
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, 'value'
    ).set;
    setter.call(input, String(value));
    if (input._valueTracker) input._valueTracker.setValue(oldValue);
    input.dispatchEvent(new Event('input', {{ bubbles: true }}));
    input.dispatchEvent(new Event('change', {{ bubbles: true }}));
  }}
  function relayoutTime(event) {{
    const candidates = [
      event['shapes[0].x0'], event['shapes[0].x1'],
      event['shapes[0].x0'], event['shapes[0].x1']
    ];
    for (const value of candidates) {{
      if (value !== undefined) {{ setTimeline(value); return; }}
    }}
  }}
  function attach() {{
    document.querySelectorAll('.js-plotly-plot').forEach((plot) => {{
      if (!plot._fullLayout || !plot._fullLayout.meta ||
          !plot._fullLayout.meta.g1_error_timeline || plot.__g1TimelineAttached) return;
      plot.__g1TimelineAttached = true;
      plot.on('plotly_click', (event) => {{
        if (event.points && event.points.length) setTimeline(event.points[0].x);
      }});
      plot.on('plotly_relayouting', relayoutTime);
      plot.on('plotly_relayout', relayoutTime);
      let graphDragging = false;
      function timelineFromMouse(event) {{
        const axis = plot._fullLayout && plot._fullLayout.xaxis;
        if (!axis || !axis.p2d) return;
        const rect = plot.getBoundingClientRect();
        const pixel = event.clientX - rect.left - axis._offset;
        setTimeline(axis.p2d(pixel));
      }}
      plot.addEventListener('mousedown', (event) => {{
        if (event.button !== 0 || event.target.closest('.modebar')) return;
        graphDragging = true;
        timelineFromMouse(event);
        event.preventDefault();
        event.stopPropagation();
      }}, true);
      window.addEventListener('mousemove', (event) => {{
        if (graphDragging) timelineFromMouse(event);
      }});
      window.addEventListener('mouseup', () => {{ graphDragging = false; }});
    }});
  }}
  window.__g1ErrorTimelineBridge = setInterval(attach, 250);
  attach();
}})();
"""
        self.server.gui._websock_interface.queue_message(
            viser_messages.RunJavascriptMessage(source=script)
        )

    def _rebuild_joint_plot(self, elapsed: float) -> None:
        if self.joint_plot_handle is None:
            return
        bag = self.bags[str(self.joint_bag.value)]
        joint_index = G1_JOINT_NAMES.index(str(self.joint_name.value))
        bag_key = str(self.joint_bag.value)
        self.joint_figure = _joint_figure(
            bag,
            joint_index,
            elapsed,
            self.common_start,
            self.clock_topics[bag_key],
        )
        self.joint_plot_handle.figure = self.joint_figure

    def update_state(self, elapsed: float, force_plots: bool = False) -> None:
        with self.lock:
            elapsed = float(np.clip(elapsed, 0.0, self.max_elapsed))
            sensor_timestamp = self.common_start + elapsed
            record_timestamps = {
                key: float(
                    bag.record_time_from_sensor_time(sensor_timestamp, self.clock_topics[key])
                )
                for key, bag in self.bags.items()
            }
            self.elapsed_number.value = round(elapsed, 4)
            self.sensor_timestamp.value = sensor_timestamp
            self.timestamp_02.value = record_timestamps["08-02"]
            self.timestamp_03.value = record_timestamps["08-03"]

            lowstate_cache = {}
            for key, bag in self.bags.items():
                lowstate_cache[key] = bag.sample_lowstate(record_timestamps[key])

            for trajectory in self.trajectories:
                bag = self.bags[trajectory.spec.bag_key]
                record_timestamp = record_timestamps[trajectory.spec.bag_key]
                active = trajectory.contains(sensor_timestamp)
                position, quaternion, text_index = trajectory.sample(sensor_timestamp)
                wxyz = quaternion_xyzw_to_wxyz(quaternion)
                frame = self.pose_frames[trajectory.name]
                marker = self.pose_markers[trajectory.name]
                frame.position = position
                frame.wxyz = wxyz
                frame.visible = active and bool(self.show_frames.value)
                marker.position = position
                marker.visible = active and bool(self.show_frames.value)

                (
                    gui_position,
                    gui_rpy,
                    gui_index,
                    gui_sample_time,
                    gui_nearest_dt,
                    gui_status,
                ) = self.pose_gui[trajectory.name]
                gui_position.value = tuple(float(value) for value in position)
                gui_rpy.value = tuple(
                    float(value) for value in Rotation.from_quat(quaternion).as_euler("xyz", degrees=True)
                )
                gui_index.value = int(text_index + 1)
                gui_sample_time.value = sensor_timestamp
                gui_nearest_dt.value = float(
                    (trajectory.times[text_index] - sensor_timestamp) * 1e3
                )
                gui_status.value = "active" if active else "outside trajectory time range"

                if trajectory.name in self.robot_models:
                    base = self.robot_bases[trajectory.name]
                    base.visible = active and bool(self.show_robot[trajectory.name].value)
                    if active:
                        pelvis_position, pelvis_quaternion = robot_pelvis_pose(
                            position, quaternion, bag, record_timestamp
                        )
                        base.position = pelvis_position
                        base.wxyz = quaternion_xyzw_to_wxyz(pelvis_quaternion)
                        q = lowstate_cache[trajectory.spec.bag_key][0]
                        model = self.robot_models[trajectory.name]
                        cfg = np.zeros(len(model.get_actuated_joint_names()), dtype=np.float64)
                        cfg[self.robot_cfg_indices[trajectory.name]] = q
                        model.update_cfg(cfg)

            inspected_key = str(self.joint_bag.value)
            inspected_bag = self.bags[inspected_key]
            inspected_timestamp = record_timestamps[inspected_key]
            q, dq, tau, imu_rpy, _, _ = inspected_bag.sample_lowstate(inspected_timestamp)
            joint_index = G1_JOINT_NAMES.index(str(self.joint_name.value))
            self.current_q.value = float(q[joint_index])
            self.current_dq.value = float(dq[joint_index])
            self.current_tau.value = float(tau[joint_index])
            self.current_imu_rpy.value = tuple(float(value) for value in imu_rpy)

            for label, (position_values, orientation_values) in self.errors.items():
                position_value = float(np.interp(elapsed, self.error_elapsed, position_values))
                orientation_value = float(
                    np.interp(elapsed, self.error_elapsed, orientation_values)
                )
                position_handle, orientation_handle = self.current_error_gui[label]
                position_handle.value = position_value
                orientation_handle.value = orientation_value

            self._update_bag_augmented_maps(elapsed)

            now = time.monotonic()
            if not self.args.no_plots and (force_plots or now - self.last_plot_update >= 0.10):
                if self.error_figure is not None and self.error_plot_handle is not None:
                    _update_error_figure(
                        self.error_figure,
                        self.error_elapsed,
                        self.errors,
                        elapsed,
                    )
                    self.error_plot_handle.figure = self.error_figure
                for trajectory in self.trajectories:
                    figure = self.pose_figures[trajectory.name]
                    _update_pose_figure(
                        figure,
                        trajectory,
                        self.common_start,
                        elapsed,
                    )
                    self.pose_plot_handles[trajectory.name].figure = figure
                if self.joint_figure is not None and self.joint_plot_handle is not None:
                    selected_bag = self.bags[str(self.joint_bag.value)]
                    selected_joint = G1_JOINT_NAMES.index(str(self.joint_name.value))
                    _update_joint_figure(
                        self.joint_figure,
                        selected_bag,
                        selected_joint,
                        elapsed,
                        self.common_start,
                        self.clock_topics[str(self.joint_bag.value)],
                    )
                    self.joint_plot_handle.figure = self.joint_figure
                self.last_plot_update = now

    def run_forever(self) -> None:
        print(f"Open http://localhost:{self.args.port} in a browser.", flush=True)
        speed_values = {"0.25×": 0.25, "0.5×": 0.5, "1×": 1.0, "2×": 2.0}
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
                    value = float(self.time_slider.value) + dt * speed_values[str(self.speed_control.value)]
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
    bags: dict[str, BagData],
    trajectories: list[Trajectory],
    points: np.ndarray,
    spatial_alignment: np.ndarray,
) -> None:
    print("\nValidation summary")
    print(f"  map points: {len(points):,}")
    for key, bag in bags.items():
        print(
            f"  bag {key}: duration={bag.duration:.3f}s, lowstate={len(bag.lowstate_times):,}, "
            f"TF pairs={sorted(bag.transforms)}, "
            f"augmented_map={len(bag.pointclouds.get('/augmented_map').times) if '/augmented_map' in bag.pointclouds else 0}"
        )
    common_start = max(trajectory.times[0] for trajectory in trajectories)
    common_end = min(trajectory.times[-1] for trajectory in trajectories)
    print(
        f"  exact common sensor interval: {common_start:.9f}..{common_end:.9f} "
        f"({common_end - common_start:.3f}s)"
    )
    for trajectory in trajectories:
        print(
            f"  {trajectory.name}: rows={len(trajectory.times):,}, "
            f"sensor time={trajectory.times[0]:.9f}..{trajectory.times[-1]:.9f}, "
            f"quaternion_order={trajectory.spec.quat_order}"
        )
        print(f"    time alignment: {trajectory.alignment_note}")
        print(f"    pose filtering: {trajectory.filter_note}")
        print(f"    spatial alignment: {trajectory.spatial_alignment_note}")
    print("  shared GLIM spatial transform:")
    print(np.array2string(spatial_alignment, precision=6, suppress_small=True))
    _, errors = _compute_error_series(trajectories, common_start, common_end)
    for label, (position_error, orientation_error) in errors.items():
        print(
            f"  {label}: position mean/max={position_error.mean():.4f}/{position_error.max():.4f} m, "
            f"orientation mean/max={orientation_error.mean():.3f}/{orientation_error.max():.3f} deg"
        )


def main() -> None:
    args = parse_args()
    bags, trajectories, points, colors, spatial_alignment = load_inputs(args)
    if args.dry_run:
        print_dry_run_summary(bags, trajectories, points, spatial_alignment)
        return
    app = ReplayApp(args, bags, trajectories, points, colors, spatial_alignment)
    app.run_forever()


if __name__ == "__main__":
    main()

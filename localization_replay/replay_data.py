"""Data loading and time-alignment utilities for the localization replay UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class TrajectorySpec:
    name: str
    path: Path
    bag_key: str
    quat_order: str
    color: tuple[int, int, int]
    match_topic: str | None = None
    time_mode: str = "raw_sensor"


@dataclass
class Trajectory:
    spec: TrajectorySpec
    raw_times: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray
    time_offset: float = 0.0
    alignment_note: str = "raw ROS timestamp"
    spatial_alignment_note: str = "identity"
    filter_note: str = "no pose filtering"

    @property
    def name(self) -> str:
        return self.spec.name

    def contains(self, timestamp: float) -> bool:
        return bool(self.times[0] <= timestamp <= self.times[-1])

    def sample(self, timestamp: float) -> tuple[np.ndarray, np.ndarray, int]:
        """Interpolate position and orientation at an absolute ROS timestamp."""
        if timestamp <= self.times[0]:
            return self.positions[0].copy(), self.quaternions_xyzw[0].copy(), 0
        if timestamp >= self.times[-1]:
            idx = len(self.times) - 1
            return self.positions[idx].copy(), self.quaternions_xyzw[idx].copy(), idx

        right = int(np.searchsorted(self.times, timestamp, side="right"))
        left = right - 1
        span = self.times[right] - self.times[left]
        alpha = 0.0 if span <= 0.0 else (timestamp - self.times[left]) / span
        position = (1.0 - alpha) * self.positions[left] + alpha * self.positions[right]

        q0 = self.quaternions_xyzw[left]
        q1 = self.quaternions_xyzw[right]
        if float(np.dot(q0, q1)) < 0.0:
            q1 = -q1
        quaternion = (1.0 - alpha) * q0 + alpha * q1
        quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
        nearest = left if alpha < 0.5 else right
        return position, quaternion, nearest


@dataclass
class TransformSeries:
    times: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray

    def sample(self, timestamp: float) -> tuple[np.ndarray, np.ndarray]:
        if len(self.times) == 1 or timestamp <= self.times[0]:
            return self.positions[0], self.quaternions_xyzw[0]
        if timestamp >= self.times[-1]:
            return self.positions[-1], self.quaternions_xyzw[-1]
        right = int(np.searchsorted(self.times, timestamp, side="right"))
        left = right - 1
        dt = self.times[right] - self.times[left]
        alpha = 0.0 if dt <= 0.0 else (timestamp - self.times[left]) / dt
        position = (1.0 - alpha) * self.positions[left] + alpha * self.positions[right]
        q0 = self.quaternions_xyzw[left]
        q1 = self.quaternions_xyzw[right]
        if float(np.dot(q0, q1)) < 0.0:
            q1 = -q1
        quaternion = (1.0 - alpha) * q0 + alpha * q1
        quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
        return position, quaternion


@dataclass
class OdometrySeries:
    times: np.ndarray
    header_times: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray

    def sample_header_time(self, timestamp: float) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate a pose using the original sensor/header clock."""
        times = self.header_times
        if timestamp <= times[0]:
            index = 0
            quaternion = self.quaternions_xyzw[index].copy()
            quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
            return self.positions[index].copy(), quaternion
        if timestamp >= times[-1]:
            index = len(times) - 1
            quaternion = self.quaternions_xyzw[index].copy()
            quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
            return self.positions[index].copy(), quaternion
        right = int(np.searchsorted(times, timestamp, side="right"))
        left = right - 1
        span = times[right] - times[left]
        alpha = 0.0 if span <= 0.0 else (timestamp - times[left]) / span
        position = (
            (1.0 - alpha) * self.positions[left] + alpha * self.positions[right]
        )
        q0 = self.quaternions_xyzw[left].copy()
        q1 = self.quaternions_xyzw[right].copy()
        q0 /= max(float(np.linalg.norm(q0)), 1e-12)
        q1 /= max(float(np.linalg.norm(q1)), 1e-12)
        if float(np.dot(q0, q1)) < 0.0:
            q1 = -q1
        quaternion = (1.0 - alpha) * q0 + alpha * q1
        quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
        return position, quaternion


@dataclass(frozen=True)
class PointCloudFrame:
    message_index: int
    recorded_time: float
    header_time: float
    frame_id: str
    points: np.ndarray


@dataclass
class PointCloudSeries:
    db_path: Path
    topic: str
    type_name: str
    message_ids: np.ndarray
    times: np.ndarray
    _cache: dict[int, PointCloudFrame] = field(default_factory=dict, repr=False)

    def frame_index_at(self, timestamp: float) -> int | None:
        """Return the newest cloud at or before timestamp (causal replay)."""
        index = int(np.searchsorted(self.times, timestamp, side="right") - 1)
        return None if index < 0 else min(index, len(self.times) - 1)

    def load_frame(self, index: int, max_points: int = 200_000) -> PointCloudFrame:
        if index in self._cache:
            return self._cache[index]
        message_id = int(self.message_ids[index])
        with _open_bag_readonly(self.db_path) as connection:
            row = connection.execute(
                "SELECT timestamp, data FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Point cloud message id {message_id} is missing from {self.db_path}")
        message = _deserialize(row[1], _message_class(self.type_name))
        points = pointcloud2_xyz(message, max_points=max_points)
        header_time = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
        frame = PointCloudFrame(
            message_index=index,
            recorded_time=int(row[0]) * 1e-9,
            header_time=header_time,
            frame_id=str(message.header.frame_id),
            points=points,
        )
        self._cache[index] = frame
        if len(self._cache) > 16:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        return frame


@dataclass
class BagData:
    key: str
    path: Path
    start_time: float
    end_time: float
    lowstate_times: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    imu_rpy: np.ndarray
    imu_quaternion_wxyz: np.ndarray
    transforms: dict[tuple[str, str], TransformSeries] = field(default_factory=dict)
    odometry: dict[str, OdometrySeries] = field(default_factory=dict)
    pointclouds: dict[str, PointCloudSeries] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def lowstate_index(self, timestamp: float) -> int:
        right = int(np.searchsorted(self.lowstate_times, timestamp))
        if right <= 0:
            return 0
        if right >= len(self.lowstate_times):
            return len(self.lowstate_times) - 1
        left = right - 1
        return left if timestamp - self.lowstate_times[left] <= self.lowstate_times[right] - timestamp else right

    def sample_lowstate(
        self, timestamp: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        idx = self.lowstate_index(timestamp)
        return (
            self.q[idx],
            self.dq[idx],
            self.tau[idx],
            self.imu_rpy[idx],
            self.imu_quaternion_wxyz[idx],
            idx,
        )

    def sensor_time_from_record_time(self, recorded_time: np.ndarray | float, clock_topic: str):
        """Map rosbag record time to the original sensor/header clock."""
        odometry = self.odometry[clock_topic]
        return np.interp(recorded_time, odometry.times, odometry.header_times)

    def record_time_from_sensor_time(self, sensor_time: np.ndarray | float, clock_topic: str):
        """Map original sensor/header time to this bag's record clock."""
        odometry = self.odometry[clock_topic]
        headers, unique_indices = np.unique(odometry.header_times, return_index=True)
        return np.interp(sensor_time, headers, odometry.times[unique_indices])

    def transform(self, parent: str, child: str, timestamp: float) -> np.ndarray | None:
        series = self.transforms.get((parent, child))
        if series is None:
            return None
        position, quaternion = series.sample(timestamp)
        return pose_matrix(position, quaternion)


def _normalize_frame(frame: str) -> str:
    return frame.strip().lstrip("/")


def pose_matrix(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    transform[:3, 3] = position
    return transform


def matrix_pose(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return transform[:3, 3].copy(), Rotation.from_matrix(transform[:3, :3]).as_quat()


def quaternion_xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    return np.asarray(quaternion)[[3, 0, 1, 2]]


def apply_spatial_transform(
    trajectory: Trajectory,
    transform: np.ndarray,
    note: str,
) -> None:
    """Apply one fixed SE(3) transform to every pose in a trajectory."""
    rotation = Rotation.from_matrix(transform[:3, :3])
    trajectory.positions = rotation.apply(trajectory.positions) + transform[:3, 3]
    trajectory.quaternions_xyzw = (
        rotation * Rotation.from_quat(trajectory.quaternions_xyzw)
    ).as_quat()
    trajectory.spatial_alignment_note = note


def pointcloud2_xyz(message, max_points: int = 200_000) -> np.ndarray:
    """Convert a sensor_msgs/PointCloud2 message to finite float32 XYZ points."""
    from sensor_msgs_py import point_cloud2

    points = point_cloud2.read_points_numpy(
        message,
        field_names=("x", "y", "z"),
        skip_nans=True,
    )
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    if max_points > 0 and len(points) > max_points:
        stride = int(np.ceil(len(points) / max_points))
        points = points[::stride]
    return points


def load_trajectory(spec: TrajectorySpec) -> Trajectory:
    if not spec.path.is_file():
        raise FileNotFoundError(f"Trajectory not found: {spec.path}")
    values = np.loadtxt(spec.path, dtype=np.float64, ndmin=2)
    if values.shape[1] < 8:
        raise ValueError(f"{spec.path} must have at least 8 columns; got {values.shape[1]}")
    values = values[:, :8]
    finite = np.all(np.isfinite(values), axis=1)
    values = values[finite]
    if len(values) < 2:
        raise ValueError(f"{spec.path} has fewer than two valid poses")
    values = values[np.argsort(values[:, 0], kind="stable")]

    raw_quaternions = values[:, 4:8]
    if spec.quat_order == "xyzw":
        quaternions = raw_quaternions.copy()
    elif spec.quat_order == "wxyz":
        quaternions = raw_quaternions[:, [1, 2, 3, 0]]
    else:
        raise ValueError(f"Unsupported quaternion order: {spec.quat_order}")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError(f"{spec.path} contains a zero quaternion")
    quaternions /= norms

    return Trajectory(
        spec=spec,
        raw_times=values[:, 0].copy(),
        times=values[:, 0].copy(),
        positions=values[:, 1:4].copy(),
        quaternions_xyzw=quaternions,
    )


def remove_trajectory_pose_matches(
    trajectory: Trajectory,
    reference: OdometrySeries,
    position_tolerance_m: float = 1.0e-5,
    orientation_tolerance_deg: float = 0.01,
) -> int:
    """Remove replayed poses that duplicate an independent odometry stream."""
    distances, indices = cKDTree(reference.positions).query(
        trajectory.positions, k=1
    )
    reference_quaternions = reference.quaternions_xyzw[indices].copy()
    reference_quaternions /= np.maximum(
        np.linalg.norm(reference_quaternions, axis=1, keepdims=True), 1.0e-12
    )
    dots = np.abs(np.einsum(
        "ij,ij->i", trajectory.quaternions_xyzw, reference_quaternions
    ))
    orientation_degrees = np.rad2deg(
        2.0 * np.arccos(np.clip(dots, 0.0, 1.0))
    )
    replayed = (
        (distances <= position_tolerance_m)
        & (orientation_degrees <= orientation_tolerance_deg)
    )
    keep = ~replayed
    removed = int(np.count_nonzero(replayed))
    if np.count_nonzero(keep) < 2:
        raise RuntimeError(
            f"Filtering replayed poses left fewer than two rows in {trajectory.name}"
        )
    trajectory.raw_times = trajectory.raw_times[keep]
    trajectory.times = trajectory.times[keep]
    trajectory.positions = trajectory.positions[keep]
    trajectory.quaternions_xyzw = trajectory.quaternions_xyzw[keep]
    trajectory.filter_note = (
        f"removed {removed} replayed poses; kept {np.count_nonzero(keep)} "
        "localization-only poses"
    )
    return removed


def remove_odometry_pose_matches(
    series: OdometrySeries,
    reference: OdometrySeries,
    position_tolerance_m: float = 1.0e-5,
    orientation_tolerance_deg: float = 0.01,
) -> tuple[OdometrySeries, int]:
    """Return odometry with exact replayed reference poses removed."""
    distances, indices = cKDTree(reference.positions).query(
        series.positions, k=1
    )
    quaternions = series.quaternions_xyzw.copy()
    quaternions /= np.maximum(
        np.linalg.norm(quaternions, axis=1, keepdims=True), 1.0e-12
    )
    reference_quaternions = reference.quaternions_xyzw[indices].copy()
    reference_quaternions /= np.maximum(
        np.linalg.norm(reference_quaternions, axis=1, keepdims=True), 1.0e-12
    )
    dots = np.abs(np.einsum("ij,ij->i", quaternions, reference_quaternions))
    orientation_degrees = np.rad2deg(
        2.0 * np.arccos(np.clip(dots, 0.0, 1.0))
    )
    replayed = (
        (distances <= position_tolerance_m)
        & (orientation_degrees <= orientation_tolerance_deg)
    )
    keep = ~replayed
    removed = int(np.count_nonzero(replayed))
    if np.count_nonzero(keep) < 2:
        raise RuntimeError(
            "Filtering replayed odometry left fewer than two poses"
        )
    return (
        OdometrySeries(
            times=series.times[keep],
            header_times=series.header_times[keep],
            positions=series.positions[keep],
            quaternions_xyzw=series.quaternions_xyzw[keep],
        ),
        removed,
    )


def _find_db3(bag_path: Path) -> Path:
    if bag_path.is_file() and bag_path.suffix == ".db3":
        return bag_path
    candidates = sorted(bag_path.glob("*.db3"))
    if not candidates:
        raise FileNotFoundError(f"No .db3 file found under {bag_path}")
    if len(candidates) > 1:
        raise ValueError(f"Split bags are not yet supported; found {len(candidates)} .db3 files in {bag_path}")
    return candidates[0]


def _open_bag_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _message_class(type_name: str):
    try:
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Python modules are unavailable. Run via ./run.sh so the Humble and Unitree overlays are sourced."
        ) from exc
    return get_message(type_name)


def _topic_info(connection: sqlite3.Connection, topic: str) -> tuple[int, str] | None:
    row = connection.execute("SELECT id, type FROM topics WHERE name = ?", (topic,)).fetchone()
    return None if row is None else (int(row[0]), str(row[1]))


def _deserialize(blob: bytes, message_class):
    from rclpy.serialization import deserialize_message

    return deserialize_message(blob, message_class)


def _load_lowstate(
    connection: sqlite3.Connection, topic: str = "/lowstate"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    info = _topic_info(connection, topic)
    if info is None:
        raise KeyError(f"Required topic {topic} not found in bag")
    topic_id, type_name = info
    message_class = _message_class(type_name)
    count = int(
        connection.execute("SELECT count(*) FROM messages WHERE topic_id = ?", (topic_id,)).fetchone()[0]
    )
    times = np.empty(count, dtype=np.float64)
    q = np.empty((count, len(G1_JOINT_NAMES)), dtype=np.float32)
    dq = np.empty_like(q)
    tau = np.empty_like(q)
    imu_rpy = np.empty((count, 3), dtype=np.float32)
    imu_quaternion_wxyz = np.empty((count, 4), dtype=np.float32)
    rows = connection.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", (topic_id,)
    )
    for index, (timestamp_ns, blob) in enumerate(rows):
        message = _deserialize(blob, message_class)
        times[index] = int(timestamp_ns) * 1e-9
        motors = message.motor_state
        q[index] = [motors[j].q for j in range(len(G1_JOINT_NAMES))]
        dq[index] = [motors[j].dq for j in range(len(G1_JOINT_NAMES))]
        tau[index] = [motors[j].tau_est for j in range(len(G1_JOINT_NAMES))]
        imu_rpy[index] = message.imu_state.rpy
        imu_quaternion_wxyz[index] = message.imu_state.quaternion
    return times, q, dq, tau, imu_rpy, imu_quaternion_wxyz


def _load_odometry(connection: sqlite3.Connection, topics: Iterable[str]) -> dict[str, OdometrySeries]:
    output: dict[str, OdometrySeries] = {}
    for topic in topics:
        info = _topic_info(connection, topic)
        if info is None:
            continue
        topic_id, type_name = info
        message_class = _message_class(type_name)
        count = int(
            connection.execute("SELECT count(*) FROM messages WHERE topic_id = ?", (topic_id,)).fetchone()[0]
        )
        times = np.empty(count, dtype=np.float64)
        header_times = np.empty(count, dtype=np.float64)
        positions = np.empty((count, 3), dtype=np.float64)
        quaternions = np.empty((count, 4), dtype=np.float64)
        rows = connection.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", (topic_id,)
        )
        for index, (timestamp_ns, blob) in enumerate(rows):
            message = _deserialize(blob, message_class)
            pose = message.pose.pose
            times[index] = int(timestamp_ns) * 1e-9
            header_times[index] = (
                float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
            )
            positions[index] = (pose.position.x, pose.position.y, pose.position.z)
            quaternions[index] = (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
        output[topic] = OdometrySeries(times, header_times, positions, quaternions)
    return output


def _load_transforms(
    connection: sqlite3.Connection,
    wanted_pairs: set[tuple[str, str]],
) -> dict[tuple[str, str], TransformSeries]:
    collected: dict[tuple[str, str], list[tuple[float, np.ndarray, np.ndarray]]] = {
        pair: [] for pair in wanted_pairs
    }
    for topic in ("/tf_static", "/tf"):
        info = _topic_info(connection, topic)
        if info is None:
            continue
        topic_id, type_name = info
        message_class = _message_class(type_name)
        rows = connection.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", (topic_id,)
        )
        for timestamp_ns, blob in rows:
            message = _deserialize(blob, message_class)
            timestamp = int(timestamp_ns) * 1e-9
            for stamped in message.transforms:
                pair = (
                    _normalize_frame(stamped.header.frame_id),
                    _normalize_frame(stamped.child_frame_id),
                )
                if pair not in collected:
                    continue
                transform = stamped.transform
                position = np.array(
                    (transform.translation.x, transform.translation.y, transform.translation.z),
                    dtype=np.float64,
                )
                quaternion = np.array(
                    (transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w),
                    dtype=np.float64,
                )
                norm = np.linalg.norm(quaternion)
                if norm > 1e-8:
                    collected[pair].append((timestamp, position, quaternion / norm))

    output: dict[tuple[str, str], TransformSeries] = {}
    for pair, samples in collected.items():
        if not samples:
            continue
        samples.sort(key=lambda sample: sample[0])
        # Static transforms can be published more than once. Keeping all samples is harmless,
        # while dynamic lidar->pelvis transforms are interpolated at playback time.
        output[pair] = TransformSeries(
            times=np.array([sample[0] for sample in samples], dtype=np.float64),
            positions=np.stack([sample[1] for sample in samples]),
            quaternions_xyzw=np.stack([sample[2] for sample in samples]),
        )
    return output


def _load_pointcloud_indices(
    connection: sqlite3.Connection,
    db_path: Path,
    topics: Iterable[str],
) -> dict[str, PointCloudSeries]:
    output: dict[str, PointCloudSeries] = {}
    for topic in topics:
        info = _topic_info(connection, topic)
        if info is None:
            continue
        topic_id, type_name = info
        rows = connection.execute(
            "SELECT id, timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (topic_id,),
        ).fetchall()
        if not rows:
            continue
        output[topic] = PointCloudSeries(
            db_path=db_path,
            topic=topic,
            type_name=type_name,
            message_ids=np.array([row[0] for row in rows], dtype=np.int64),
            times=np.array([int(row[1]) * 1e-9 for row in rows], dtype=np.float64),
        )
    return output


def load_bag(
    key: str,
    bag_path: Path,
    odometry_topics: Iterable[str] = (),
    pointcloud_topics: Iterable[str] = (),
) -> BagData:
    db_path = _find_db3(bag_path)
    with _open_bag_readonly(db_path) as connection:
        bounds = connection.execute("SELECT min(timestamp), max(timestamp) FROM messages").fetchone()
        if bounds is None or bounds[0] is None:
            raise ValueError(f"Bag has no messages: {db_path}")
        start_time = int(bounds[0]) * 1e-9
        end_time = int(bounds[1]) * 1e-9
        lowstate = _load_lowstate(connection)
        odometry = _load_odometry(connection, odometry_topics)
        transforms = _load_transforms(
            connection,
            wanted_pairs={
                ("map", "world"),
                ("imu", "lidar"),
                ("lidar", "pelvis"),
                ("mid360_link", "mid360_link_frame"),
                ("mid360_link_frame", "mid360_link_IMU"),
            },
        )
        pointclouds = _load_pointcloud_indices(connection, db_path, pointcloud_topics)
    return BagData(
        key=key,
        path=bag_path,
        start_time=start_time,
        end_time=end_time,
        lowstate_times=lowstate[0],
        q=lowstate[1],
        dq=lowstate[2],
        tau=lowstate[3],
        imu_rpy=lowstate[4],
        imu_quaternion_wxyz=lowstate[5],
        transforms=transforms,
        odometry=odometry,
        pointclouds=pointclouds,
    )


def load_odometry_only(
    bag_path: Path,
    topics: Iterable[str],
) -> dict[str, OdometrySeries]:
    """Load only odometry from a bag, avoiding lowstate/TF replay overhead."""
    db_path = _find_db3(bag_path)
    with _open_bag_readonly(db_path) as connection:
        return _load_odometry(connection, topics)


def align_trajectory_time(
    trajectory: Trajectory,
    bag: BagData,
    forced_offset: float | None = None,
    odometry_override: OdometrySeries | None = None,
) -> None:
    """Convert a trajectory time column to the shared original sensor clock."""
    mode = trajectory.spec.time_mode
    if mode == "raw_sensor":
        trajectory.time_offset = 0.0
        trajectory.times = trajectory.raw_times.copy()
        trajectory.alignment_note = "raw timestamp is original sensor/header time"
        return

    if forced_offset is not None:
        offset = float(forced_offset)
        trajectory.times = trajectory.raw_times + offset
        trajectory.time_offset = offset
        trajectory.alignment_note = f"forced raw -> sensor offset {offset:+.6f} s"
        return

    if odometry_override is None and trajectory.spec.match_topic not in bag.odometry:
        raise KeyError(
            f"{trajectory.name} requires {trajectory.spec.match_topic} for {mode} time conversion"
        )
    odometry = (
        odometry_override
        if odometry_override is not None
        else bag.odometry[trajectory.spec.match_topic]
    )

    if mode == "record_to_sensor":
        mapped = np.interp(trajectory.raw_times, odometry.times, odometry.header_times)
        offsets = mapped - trajectory.raw_times
        trajectory.times = mapped
        trajectory.time_offset = float(np.median(offsets))
        trajectory.alignment_note = (
            f"mapped {trajectory.spec.match_topic} bag record time -> sensor header time; "
            f"median {trajectory.time_offset:+.6f} s"
        )
        return

    if mode != "pose_match_sensor":
        raise ValueError(f"Unsupported trajectory time mode: {mode}")
    sample_indices = np.linspace(
        0, len(trajectory.positions) - 1, min(300, len(trajectory.positions)), dtype=int
    )
    distances, odom_indices = cKDTree(odometry.positions).query(
        trajectory.positions[sample_indices], k=1
    )
    robust_limit = max(0.02, float(np.percentile(distances, 25)) * 5.0)
    good = distances <= robust_limit
    if int(np.count_nonzero(good)) < 10:
        raise RuntimeError(
            f"Could not align {trajectory.name}: only {np.count_nonzero(good)} pose matches on "
            f"{trajectory.spec.match_topic}"
        )
    target_sensor_times = odometry.header_times[odom_indices[good]]
    offsets = target_sensor_times - trajectory.raw_times[sample_indices[good]]
    offset = float(np.median(offsets))
    residual_ms = float(np.median(np.abs(offsets - offset)) * 1e3)
    position_error = float(np.median(distances[good]))
    trajectory.time_offset = offset
    trajectory.times = trajectory.raw_times + offset
    trajectory.alignment_note = (
        f"pose-matched {trajectory.spec.match_topic} raw clock -> sensor header clock: "
        f"{offset:+.6f} s (median |dt residual| {residual_ms:.2f} ms, "
        f"position error {position_error:.4f} m)"
    )


def robot_pelvis_pose(
    imu_position: np.ndarray,
    imu_quaternion_xyzw: np.ndarray,
    bag: BagData,
    timestamp: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose map->imu, imu->lidar, and lidar->pelvis at playback time."""
    world_from_imu = pose_matrix(imu_position, imu_quaternion_xyzw)
    imu_from_lidar = bag.transform("imu", "lidar", timestamp)
    lidar_from_pelvis = bag.transform("lidar", "pelvis", timestamp)
    if imu_from_lidar is None:
        imu_from_lidar = np.eye(4)
    if lidar_from_pelvis is None:
        lidar_from_pelvis = np.eye(4)
    return matrix_pose(world_from_imu @ imu_from_lidar @ lidar_from_pelvis)


_PCD_TYPE_MAP: dict[tuple[str, int], str] = {
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    ("I", 1): "i1",
    ("I", 2): "<i2",
    ("I", 4): "<i4",
    ("I", 8): "<i8",
    ("U", 1): "u1",
    ("U", 2): "<u2",
    ("U", 4): "<u4",
    ("U", 8): "<u8",
}


def load_pcd(path: Path, max_points: int = 1_000_000) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Load binary or ASCII PCD points and RGB/intensity colors without Open3D."""
    if not path.is_file():
        raise FileNotFoundError(f"PCD not found: {path}")
    header: dict[str, str] = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Malformed PCD header: {path}")
            decoded = line.decode("ascii", errors="strict").strip()
            if not decoded or decoded.startswith("#"):
                continue
            key, *rest = decoded.split()
            header[key.upper()] = " ".join(rest)
            if key.upper() == "DATA":
                break

        fields = header["FIELDS"].split()
        sizes = [int(value) for value in header["SIZE"].split()]
        types = header["TYPE"].split()
        counts = [int(value) for value in header.get("COUNT", " ".join("1" for _ in fields)).split()]
        point_count = int(header.get("POINTS", header.get("WIDTH", "0")))
        data_kind = header["DATA"].lower()

        if data_kind == "binary":
            dtype_fields: list[tuple] = []
            for name, size, kind, count in zip(fields, sizes, types, counts):
                scalar_type = _PCD_TYPE_MAP.get((kind.upper(), size))
                if scalar_type is None:
                    raise ValueError(f"Unsupported PCD field type: {kind}{size}")
                dtype_fields.append((name, scalar_type) if count == 1 else (name, scalar_type, (count,)))
            records = np.fromfile(stream, dtype=np.dtype(dtype_fields), count=point_count)
        elif data_kind == "ascii":
            raw = np.loadtxt(stream, dtype=np.float64, ndmin=2)
            expanded_names: list[str] = []
            for name, count in zip(fields, counts):
                expanded_names.extend([name] if count == 1 else [f"{name}_{i}" for i in range(count)])
            records = {name: raw[:, index] for index, name in enumerate(expanded_names)}
        else:
            raise ValueError(f"Unsupported PCD DATA mode: {data_kind}")

    def field(name: str) -> np.ndarray | None:
        if isinstance(records, dict):
            return records.get(name)
        return None if records.dtype.names is None or name not in records.dtype.names else records[name]

    xyz_fields = [field(axis) for axis in ("x", "y", "z")]
    if any(value is None for value in xyz_fields):
        raise ValueError("PCD must contain x, y, z fields")
    points = np.column_stack(xyz_fields).astype(np.float32, copy=False)
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]

    rgb = field("rgb")
    rgba = field("rgba")
    packed_field = rgba if rgba is not None else rgb
    colors: np.ndarray | None = None
    if packed_field is not None:
        packed_raw = np.asarray(packed_field)[finite]
        if packed_raw.dtype.kind == "f" and packed_raw.dtype.itemsize == 4:
            packed = packed_raw.view(np.uint32)
        else:
            packed = packed_raw.astype(np.uint32, copy=False)
        candidate = np.column_stack(
            ((packed >> 16) & 255, (packed >> 8) & 255, packed & 255)
        ).astype(np.uint8)
        if np.count_nonzero(candidate) > max(30, len(candidate) // 100):
            colors = candidate

    if colors is None:
        intensity = field("intensity")
        if intensity is not None:
            values = np.asarray(intensity, dtype=np.float32)[finite]
        else:
            values = points[:, 2]
        low, high = np.nanpercentile(values, (2.0, 98.0))
        normalized = np.clip((values - low) / max(float(high - low), 1e-6), 0.0, 1.0)
        colors = np.column_stack(
            (
                45.0 + 205.0 * normalized,
                80.0 + 150.0 * (1.0 - np.abs(2.0 * normalized - 1.0)),
                235.0 - 180.0 * normalized,
            )
        ).astype(np.uint8)

    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(0)
        selected = np.sort(rng.choice(len(points), size=max_points, replace=False))
        points = points[selected]
        colors = colors[selected]
    return points, colors, header

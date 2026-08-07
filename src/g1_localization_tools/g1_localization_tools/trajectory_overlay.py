import bisect
import copy
import hashlib
import json
import math
from collections import OrderedDict, deque

import numpy as np
import rclpy
import rosbag2_py
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from livox_ros_driver2.msg import CustomMsg
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float64, String
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


def stamp_nanoseconds(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def sample_signature(stamp_ns, position, quaternion):
    # Re-recording preserves these doubles exactly. Rounding only absorbs
    # serialization noise while retaining the sensor timestamp in the key.
    values = (*position, *quaternion)
    return (stamp_ns, *(round(float(value), 9) for value in values))


def pointcloud_signature(message):
    return (
        stamp_nanoseconds(message.header.stamp),
        hashlib.sha256(bytes(message.data)).digest(),
    )


def read_bag_pointclouds(bag_path, topic):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', ''),
    )
    available = {
        item.name: item.type for item in reader.get_all_topics_and_types()}
    if available.get(topic) != 'sensor_msgs/msg/PointCloud2':
        raise RuntimeError(
            f'{bag_path}: missing sensor_msgs/msg/PointCloud2 topic {topic}')
    messages = []
    while reader.has_next():
        current_topic, serialized, _ = reader.read_next()
        if current_topic == topic:
            messages.append(deserialize_message(serialized, PointCloud2))
    return messages


def read_bag_odometry(bag_path, topics):
    requested = set(topics)
    samples = {topic: [] for topic in requested}
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', ''),
    )
    available = {
        item.name: item.type for item in reader.get_all_topics_and_types()}
    for topic in requested:
        if available.get(topic) != 'nav_msgs/msg/Odometry':
            raise RuntimeError(
                f'{bag_path}: missing nav_msgs/msg/Odometry topic {topic}')

    while reader.has_next():
        topic, serialized, _ = reader.read_next()
        if topic not in requested:
            continue
        message = deserialize_message(serialized, Odometry)
        pose = message.pose.pose
        samples[topic].append({
            'stamp_ns': stamp_nanoseconds(message.header.stamp),
            'position': np.asarray([
                pose.position.x, pose.position.y, pose.position.z],
                dtype=np.float64),
            'quaternion': np.asarray([
                pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w],
                dtype=np.float64),
        })

    for topic, records in samples.items():
        if len(records) < 2:
            raise RuntimeError(f'{bag_path}:{topic}: fewer than two poses')
        records.sort(key=lambda record: record['stamp_ns'])
    return samples


def normalized_quaternion(quaternion):
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1.0e-12:
        raise RuntimeError('zero or non-finite quaternion in odometry')
    return quaternion / norm


def slerp(left, right, ratio):
    left = normalized_quaternion(left)
    right = normalized_quaternion(right)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return normalized_quaternion(left + ratio * (right - left))
    angle = math.acos(dot)
    denominator = math.sin(angle)
    return (
        math.sin((1.0 - ratio) * angle) * left +
        math.sin(ratio * angle) * right
    ) / denominator


def interpolate_sample(records, stamp_ns, max_gap_ns=None):
    stamps = [record['stamp_ns'] for record in records]
    index = bisect.bisect_left(stamps, stamp_ns)
    if index < len(records) and stamps[index] == stamp_ns:
        return records[index]
    if index == 0 or index == len(records):
        return None
    before = records[index - 1]
    after = records[index]
    gap = after['stamp_ns'] - before['stamp_ns']
    if gap <= 0 or (max_gap_ns is not None and gap > max_gap_ns):
        return None
    ratio = (stamp_ns - before['stamp_ns']) / gap
    return {
        'stamp_ns': stamp_ns,
        'position': (
            before['position'] + ratio *
            (after['position'] - before['position'])),
        'quaternion': slerp(
            before['quaternion'], after['quaternion'], ratio),
    }


def pose_matrix(sample):
    x, y, z, w = normalized_quaternion(sample['quaternion'])
    rotation = np.asarray([
        [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w),
         2.0 * (x*z + y*w)],
        [2.0 * (x*y + z*w), 1.0 - 2.0 * (x*x + z*z),
         2.0 * (y*z - x*w)],
        [2.0 * (x*z - y*w), 2.0 * (y*z + x*w),
         1.0 - 2.0 * (x*x + y*y)],
    ], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = sample['position']
    return transform


def quaternion_multiply(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray([
        lw*rx + lx*rw + ly*rz - lz*ry,
        lw*ry - lx*rz + ly*rw + lz*rx,
        lw*rz + lx*ry - ly*rx + lz*rw,
        lw*rw - lx*rx - ly*ry - lz*rz,
    ], dtype=np.float64)


def quaternion_conjugate(quaternion):
    x, y, z, w = normalized_quaternion(quaternion)
    return np.asarray([-x, -y, -z, w], dtype=np.float64)


def xyz_cloud_message(stamp, frame, points):
    values = np.asarray(points, dtype='<f4').reshape((-1, 3))
    message = PointCloud2()
    message.header.stamp = stamp
    message.header.frame_id = frame
    message.height = 1
    message.width = len(values)
    message.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 12
    message.row_step = 12 * len(values)
    message.data = values.tobytes()
    message.is_dense = bool(np.isfinite(values).all())
    return message


class TrajectoryOverlay(Node):
    def __init__(self):
        super().__init__('glim_super_lio_trajectory_overlay')
        self.declare_parameter('reference_bag_path', '')
        self.declare_parameter(
            'reference_topic', '/glim_ros/odom_corrected')
        self.declare_parameter(
            'calibration_reference_topic', '/odometry/odom')
        self.declare_parameter('map_frame_bag_path', '')
        self.declare_parameter('map_frame_topic', '/odometry/odom')
        self.declare_parameter('replayed_input_topic', '/lio/odom')
        self.declare_parameter('estimated_topic', '/comparison/input_odom')
        self.declare_parameter(
            'clean_estimated_topic', '/comparison/super_lio_localization_odom')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('path_sample_period_sec', 0.03)
        self.declare_parameter('calibration_reference_index', 0)
        self.declare_parameter('calibration_max_gap_sec', 0.10)
        self.declare_parameter('glim_cloud_topic', '/augmented_map')
        self.declare_parameter(
            'raw_lidar_topic', '/comparison/input_lidar')
        self.declare_parameter('glim_cloud_frame', 'comparison_glim_map')
        self.declare_parameter('super_cloud_scan_stride', 8)
        self.declare_parameter('super_cloud_voxel_size_m', 0.08)
        self.declare_parameter('super_cloud_publish_period_sec', 0.5)
        self.declare_parameter('super_cloud_max_points', 400000)
        self.declare_parameter(
            'lidar_to_imu_translation', [-0.011, -0.02329, 0.04412])

        reference_bag = self.get_parameter('reference_bag_path').value
        map_frame_bag = self.get_parameter('map_frame_bag_path').value
        if not reference_bag or not map_frame_bag:
            raise RuntimeError(
                'reference_bag_path and map_frame_bag_path are required')

        reference_topic = self.get_parameter('reference_topic').value
        calibration_reference_topic = self.get_parameter(
            'calibration_reference_topic').value
        map_frame_topic = self.get_parameter('map_frame_topic').value
        replayed_input_topic = self.get_parameter(
            'replayed_input_topic').value

        reference_data = read_bag_odometry(
            reference_bag,
            (reference_topic, calibration_reference_topic))
        map_data = read_bag_odometry(
            map_frame_bag, (map_frame_topic, replayed_input_topic))
        self._reference_samples = reference_data[reference_topic]

        calibration_samples = reference_data[calibration_reference_topic]
        anchor_index = int(
            self.get_parameter('calibration_reference_index').value)
        if not 0 <= anchor_index < len(calibration_samples):
            raise RuntimeError('calibration_reference_index is out of range')
        reference_anchor = calibration_samples[anchor_index]
        max_gap_ns = int(float(
            self.get_parameter('calibration_max_gap_sec').value) * 1.0e9)
        map_anchor = interpolate_sample(
            map_data[map_frame_topic], reference_anchor['stamp_ns'],
            max_gap_ns=max_gap_ns)
        if map_anchor is None:
            raise RuntimeError(
                'map-frame calibration pose does not bracket the selected '
                'GLIM sensor timestamp')

        # This is a one-pose frame calibration, not a trajectory fit:
        # T_map_glim = T_map_sensor(anchor) * inv(T_glim_sensor(anchor)).
        self._map_from_glim = (
            pose_matrix(map_anchor) @ np.linalg.inv(pose_matrix(reference_anchor)))
        self._transform_source = (
            f'one pose at sensor stamp '
            f'{reference_anchor["stamp_ns"] * 1.0e-9:.9f}')

        replayed_records = map_data[replayed_input_topic]
        self._replayed_signatures = {
            sample_signature(
                record['stamp_ns'], record['position'], record['quaternion'])
            for record in replayed_records
        }

        glim_cloud_topic = self.get_parameter('glim_cloud_topic').value
        replayed_cloud_signatures = {
            pointcloud_signature(message)
            for message in read_bag_pointclouds(
                map_frame_bag, glim_cloud_topic)
        }
        self._glim_clouds = [
            message for message in read_bag_pointclouds(
                reference_bag, glim_cloud_topic)
            if pointcloud_signature(message) not in replayed_cloud_signatures
        ]
        self._glim_clouds.sort(
            key=lambda message: stamp_nanoseconds(message.header.stamp))
        self._glim_cloud_stamps = [
            stamp_nanoseconds(message.header.stamp)
            for message in self._glim_clouds
        ]
        if not self._glim_clouds:
            raise RuntimeError(
                'No GLIM-generated point clouds remained after filtering '
                'replayed input clouds')

        self._world_frame = self.get_parameter('world_frame').value
        self._glim_cloud_frame = self.get_parameter(
            'glim_cloud_frame').value
        self._path_sample_period = float(
            self.get_parameter('path_sample_period_sec').value)
        self._super_cloud_scan_stride = max(
            1, int(self.get_parameter('super_cloud_scan_stride').value))
        self._super_cloud_voxel_size = float(
            self.get_parameter('super_cloud_voxel_size_m').value)
        if self._super_cloud_voxel_size <= 0.0:
            raise RuntimeError('super_cloud_voxel_size_m must be positive')
        self._super_cloud_publish_period_ns = int(float(
            self.get_parameter('super_cloud_publish_period_sec').value) *
            1.0e9)
        self._super_cloud_max_points = max(
            1, int(self.get_parameter('super_cloud_max_points').value))
        lidar_to_imu = list(
            self.get_parameter('lidar_to_imu_translation').value)
        if len(lidar_to_imu) != 3:
            raise RuntimeError('lidar_to_imu_translation must have 3 values')
        self._lidar_to_imu = np.asarray(lidar_to_imu, dtype=np.float64)

        transient_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=200,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._super_path_publisher = self.create_publisher(
            Path, '/comparison/super_lio_path', transient_qos)
        self._glim_path_publisher = self.create_publisher(
            Path, '/comparison/glim_path_in_super_world', transient_qos)
        self._marker_publisher = self.create_publisher(
            MarkerArray, '/comparison/markers', transient_qos)
        self._error_publisher = self.create_publisher(
            Float64, '/comparison/current_position_error_m', 10)
        self._status_publisher = self.create_publisher(
            String, '/comparison/status', 10)
        self._glim_cloud_publisher = self.create_publisher(
            PointCloud2, '/comparison/glim_live_cloud', transient_qos)
        self._super_cloud_publisher = self.create_publisher(
            PointCloud2, '/comparison/super_lio_registered_cloud',
            transient_qos)
        self._clean_estimate_publisher = self.create_publisher(
            Odometry,
            self.get_parameter('clean_estimated_topic').value,
            input_qos)
        self.create_subscription(
            Odometry, self.get_parameter('estimated_topic').value,
            self._estimated_callback, input_qos)
        self.create_subscription(
            CustomMsg, self.get_parameter('raw_lidar_topic').value,
            self._lidar_callback, input_qos)

        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        fixed_transform = TransformStamped()
        fixed_transform.header.stamp = self.get_clock().now().to_msg()
        fixed_transform.header.frame_id = self._world_frame
        fixed_transform.child_frame_id = self._glim_cloud_frame
        fixed_transform.transform.translation.x = float(
            self._map_from_glim[0, 3])
        fixed_transform.transform.translation.y = float(
            self._map_from_glim[1, 3])
        fixed_transform.transform.translation.z = float(
            self._map_from_glim[2, 3])
        map_from_glim_quaternion = normalized_quaternion(
            quaternion_multiply(
                map_anchor['quaternion'],
                quaternion_conjugate(reference_anchor['quaternion'])))
        fixed_transform.transform.rotation.x = float(
            map_from_glim_quaternion[0])
        fixed_transform.transform.rotation.y = float(
            map_from_glim_quaternion[1])
        fixed_transform.transform.rotation.z = float(
            map_from_glim_quaternion[2])
        fixed_transform.transform.rotation.w = float(
            map_from_glim_quaternion[3])
        self._static_tf_broadcaster.sendTransform(fixed_transform)

        self._super_path = Path()
        self._glim_path = Path()
        self._super_path.header.frame_id = self._world_frame
        self._glim_path.header.frame_id = self._world_frame
        self._first_estimated_stamp_ns = None
        self._last_estimated_stamp_ns = None
        self._last_path_elapsed = None
        self._errors = []
        self._last_log_second = -1
        self._filtered_replayed_count = 0
        self._lidar_scans = deque(maxlen=24)
        self._last_used_lidar_stamp_ns = None
        self._super_cloud_voxels = OrderedDict()
        self._last_super_cloud_publish_stamp_ns = None
        self._last_glim_cloud_index = None
        self._glim_cloud_publish_logged = False
        self._super_cloud_publish_logged = False

        matrix_text = np.array2string(
            self._map_from_glim, precision=9, suppress_small=True)
        self.get_logger().info(
            f'Loaded {len(self._reference_samples)} GLIM poses from '
            f'{reference_bag}:{reference_topic}')
        self.get_logger().info(
            f'Using fixed T_map_glim from {self._transform_source}; '
            f'no trajectory alignment or fitted time offset:\n{matrix_text}')
        self.get_logger().info(
            f'Will exclude {len(self._replayed_signatures)} original '
            f'{replayed_input_topic} poses mixed into the evaluation bag')
        self.get_logger().info(
            f'Loaded {len(self._glim_clouds)} GLIM-generated point clouds; '
            'Super-LIO cloud is reconstructed from recorded LiDAR scans and '
            'the cleaned localization poses')

    def _glim_to_world(self, position):
        homogeneous = np.asarray(
            [position[0], position[1], position[2], 1.0],
            dtype=np.float64)
        return tuple((self._map_from_glim @ homogeneous)[:3])

    def _lidar_callback(self, message):
        points = np.asarray([
            (message.points[index].x,
             message.points[index].y,
             message.points[index].z)
            for index in range(
                0, len(message.points), self._super_cloud_scan_stride)
        ], dtype=np.float64)
        if len(points) == 0:
            return
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return
        self._lidar_scans.append((
            stamp_nanoseconds(message.header.stamp), points))

    def _publish_glim_cloud(self, stamp_ns, output_stamp):
        index = bisect.bisect_right(self._glim_cloud_stamps, stamp_ns) - 1
        if index < 0 or index == self._last_glim_cloud_index:
            return
        if stamp_ns - self._glim_cloud_stamps[index] > 1_000_000_000:
            return
        message = copy.deepcopy(self._glim_clouds[index])
        message.header.stamp = output_stamp
        message.header.frame_id = self._glim_cloud_frame
        self._glim_cloud_publisher.publish(message)
        self._last_glim_cloud_index = index
        if not self._glim_cloud_publish_logged:
            self._glim_cloud_publish_logged = True
            self.get_logger().info(
                f'Publishing GLIM cloud with {message.width * message.height} '
                f'points in {self._glim_cloud_frame}')

    def _accumulate_super_cloud(self, odometry, stamp_ns):
        if not self._lidar_scans:
            return
        lidar_stamp_ns, lidar_points = min(
            self._lidar_scans, key=lambda item: abs(item[0] - stamp_ns))
        if (abs(lidar_stamp_ns - stamp_ns) > 50_000_000 or
                lidar_stamp_ns == self._last_used_lidar_stamp_ns):
            return
        self._last_used_lidar_stamp_ns = lidar_stamp_ns

        pose = odometry.pose.pose
        world_from_imu = pose_matrix({
            'position': np.asarray([
                pose.position.x, pose.position.y, pose.position.z],
                dtype=np.float64),
            'quaternion': np.asarray([
                pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w],
                dtype=np.float64),
        })
        points_imu = lidar_points + self._lidar_to_imu
        points_world = (
            points_imu @ world_from_imu[:3, :3].T +
            world_from_imu[:3, 3])
        voxel_keys = np.floor(
            points_world / self._super_cloud_voxel_size).astype(np.int64)
        _, unique_indices = np.unique(
            voxel_keys, axis=0, return_index=True)
        for index in unique_indices:
            key = tuple(int(value) for value in voxel_keys[index])
            if key not in self._super_cloud_voxels:
                self._super_cloud_voxels[key] = points_world[index]
        while len(self._super_cloud_voxels) > self._super_cloud_max_points:
            self._super_cloud_voxels.popitem(last=False)

        should_publish = (
            self._last_super_cloud_publish_stamp_ns is None or
            stamp_ns - self._last_super_cloud_publish_stamp_ns >=
            self._super_cloud_publish_period_ns)
        if should_publish:
            cloud = xyz_cloud_message(
                odometry.header.stamp, self._world_frame,
                list(self._super_cloud_voxels.values()))
            self._super_cloud_publisher.publish(cloud)
            self._last_super_cloud_publish_stamp_ns = stamp_ns
            if not self._super_cloud_publish_logged:
                self._super_cloud_publish_logged = True
                self.get_logger().info(
                    f'Publishing reconstructed Super-LIO cloud with '
                    f'{cloud.width} accumulated voxel points in '
                    f'{self._world_frame}')

    def _clear(self):
        self._super_path.poses.clear()
        self._glim_path.poses.clear()
        self._errors.clear()
        self._last_path_elapsed = None
        self._last_log_second = -1
        self._last_glim_cloud_index = None
        self._last_used_lidar_stamp_ns = None
        self._last_super_cloud_publish_stamp_ns = None
        self._super_cloud_voxels.clear()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        self._marker_publisher.publish(MarkerArray(markers=[delete_all]))

    def _pose_stamped(self, stamp, position, orientation=None):
        pose = PoseStamped()
        pose.header.frame_id = self._world_frame
        pose.header.stamp = stamp
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])
        if orientation is None:
            pose.pose.orientation.w = 1.0
        else:
            pose.pose.orientation = copy.deepcopy(orientation)
        return pose

    @staticmethod
    def _sphere_marker(marker_id, stamp, frame, position, color):
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = stamp
        marker.ns = 'trajectory_comparison_pose'
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.18
        marker.scale.y = 0.18
        marker.scale.z = 0.18
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        return marker

    def _publish_markers(self, stamp, super_position, glim_position,
                         current_error, rmse, p95):
        markers = [
            self._sphere_marker(
                0, stamp, self._world_frame, super_position,
                (0.05, 1.0, 0.1, 1.0)),
            self._sphere_marker(
                1, stamp, self._world_frame, glim_position,
                (1.0, 0.05, 1.0, 1.0)),
        ]
        line = Marker()
        line.header.frame_id = self._world_frame
        line.header.stamp = stamp
        line.ns = 'trajectory_comparison_error'
        line.id = 2
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.025
        line.points = [
            Point(x=float(super_position[0]), y=float(super_position[1]),
                  z=float(super_position[2])),
            Point(x=float(glim_position[0]), y=float(glim_position[1]),
                  z=float(glim_position[2])),
        ]
        if current_error < 0.05:
            line.color.r, line.color.g, line.color.b = 0.1, 1.0, 0.1
        elif current_error < 0.10:
            line.color.r, line.color.g, line.color.b = 1.0, 0.8, 0.0
        else:
            line.color.r, line.color.g, line.color.b = 1.0, 0.1, 0.1
        line.color.a = 1.0
        markers.append(line)

        text = Marker()
        text.header.frame_id = self._world_frame
        text.header.stamp = stamp
        text.ns = 'trajectory_comparison_error'
        text.id = 3
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(
            (super_position[0] + glim_position[0]) * 0.5)
        text.pose.position.y = float(
            (super_position[1] + glim_position[1]) * 0.5)
        text.pose.position.z = float(
            max(super_position[2], glim_position[2]) + 0.55)
        text.pose.orientation.w = 1.0
        text.scale.z = 0.22
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0
        text.text = (
            f'now {current_error * 100.0:.1f} cm  |  '
            f'RMSE {rmse * 100.0:.1f} cm  |  p95 {p95 * 100.0:.1f} cm')
        markers.append(text)
        self._marker_publisher.publish(MarkerArray(markers=markers))

    def _is_replayed_input(self, message):
        pose = message.pose.pose
        position = (
            pose.position.x, pose.position.y, pose.position.z)
        quaternion = (
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w)
        return sample_signature(
            stamp_nanoseconds(message.header.stamp),
            position, quaternion) in self._replayed_signatures

    def _estimated_callback(self, message):
        # The recorded comparison bag contains both the replayed input
        # /lio/odom and the newly computed relocation output on one topic.
        # Only the latter is an evaluated Super-LIO localization result.
        if self._is_replayed_input(message):
            self._filtered_replayed_count += 1
            return

        stamp_ns = stamp_nanoseconds(message.header.stamp)
        if (self._last_estimated_stamp_ns is not None and
                stamp_ns < self._last_estimated_stamp_ns - 500_000_000):
            self.get_logger().info(
                'Detected bag loop by sensor timestamp; resetting path trails')
            self._first_estimated_stamp_ns = None
            self._clear()
        self._last_estimated_stamp_ns = stamp_ns
        if self._first_estimated_stamp_ns is None:
            self._first_estimated_stamp_ns = stamp_ns
        elapsed = (stamp_ns - self._first_estimated_stamp_ns) * 1.0e-9

        reference = interpolate_sample(
            self._reference_samples, stamp_ns,
            max_gap_ns=100_000_000)
        if reference is None:
            return

        glim_position = self._glim_to_world(reference['position'])
        estimate = message.pose.pose.position
        super_position = (estimate.x, estimate.y, estimate.z)
        self._publish_glim_cloud(stamp_ns, message.header.stamp)
        self._accumulate_super_cloud(message, stamp_ns)
        error = math.sqrt(sum(
            (super_position[axis] - glim_position[axis]) ** 2
            for axis in range(3)))
        self._errors.append(error)
        rmse = math.sqrt(sum(value * value for value in self._errors) /
                         len(self._errors))
        sorted_errors = sorted(self._errors)
        p95_index = min(len(sorted_errors) - 1,
                        int(math.ceil(0.95 * len(sorted_errors))) - 1)
        p95 = sorted_errors[p95_index]

        if (self._last_path_elapsed is None or
                elapsed - self._last_path_elapsed >= self._path_sample_period):
            self._super_path.poses.append(self._pose_stamped(
                message.header.stamp, super_position,
                message.pose.pose.orientation))
            self._glim_path.poses.append(self._pose_stamped(
                message.header.stamp, glim_position))
            self._last_path_elapsed = elapsed

        self._super_path.header.stamp = message.header.stamp
        self._glim_path.header.stamp = message.header.stamp
        self._super_path_publisher.publish(self._super_path)
        self._glim_path_publisher.publish(self._glim_path)
        self._publish_markers(
            message.header.stamp, super_position, glim_position,
            error, rmse, p95)
        self._error_publisher.publish(Float64(data=error))
        clean_message = copy.deepcopy(message)
        clean_message.header.frame_id = self._world_frame
        self._clean_estimate_publisher.publish(clean_message)
        status = {
            'elapsed_sec': elapsed,
            'sensor_stamp_sec': stamp_ns * 1.0e-9,
            'current_error_m': error,
            'rmse_m': rmse,
            'p95_m': p95,
            'matched_count': len(self._errors),
            'filtered_replayed_count': self._filtered_replayed_count,
            'glim_cloud_index': self._last_glim_cloud_index,
            'super_cloud_points': len(self._super_cloud_voxels),
            'alignment': 'fixed_one_pose_frame_transform',
        }
        self._status_publisher.publish(String(data=json.dumps(status)))

        log_second = int(elapsed)
        if log_second > self._last_log_second:
            self._last_log_second = log_second
            self.get_logger().info(
                f't={elapsed:.1f}s error={error * 100.0:.1f}cm '
                f'RMSE={rmse * 100.0:.1f}cm p95={p95 * 100.0:.1f}cm')


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryOverlay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RuntimeError):
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, RuntimeError):
            pass


if __name__ == '__main__':
    main()

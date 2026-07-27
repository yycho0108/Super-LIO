import bisect
import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rclpy
from livox_ros_driver2.msg import CustomMsg
from nav_msgs.msg import Odometry, Path as PathMessage
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial import cKDTree
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2

from .common import (
    finite_odometry,
    quaternion_yaw,
    read_binary_pcd_xyz,
    stamp_seconds,
    wrapped_angle,
)


POSE_TOPICS = {
    '/lio/odom': 'lio_odom',
    '/lio/robo/odom': 'lio_robo_odom',
    '/lio/imu/odom': 'lio_imu_odom',
    '/localization/policy_odom_50hz': 'policy_odom_50hz',
    '/reference/glim_odom': 'reference_glim_odom',
}


class LocalizationEvaluator(Node):
    def __init__(self):
        super().__init__('localization_evaluator')
        self.declare_parameter('output_dir', '/tmp/super_lio_evaluation')
        self.declare_parameter('playback_rate', 1.0)
        self.declare_parameter('map_path', '')
        self.declare_parameter('cloud_stride', 10)
        self._output_dir = Path(self.get_parameter('output_dir').value)
        self._playback_rate = float(self.get_parameter('playback_rate').value)
        self._cloud_stride = max(1, int(self.get_parameter('cloud_stride').value))
        self._start_monotonic = time.monotonic()
        self._written = False
        self._pose_records = defaultdict(list)
        self._sensor_records = defaultdict(list)
        self._total_counts = defaultdict(int)
        self._nonfinite_counts = defaultdict(int)
        self._cloud_callbacks = 0
        self._cloud_distances = []

        map_path = self.get_parameter('map_path').value
        self._map_tree = None
        if map_path:
            map_xyz = read_binary_pcd_xyz(map_path)
            self._map_tree = cKDTree(map_xyz)
            self.get_logger().info(
                f'Loaded {len(map_xyz)} map points for cloud overlap checks')

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=200,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        for topic, key in POSE_TOPICS.items():
            self.create_subscription(
                Odometry, topic,
                lambda message, name=key: self._pose_callback(name, message), qos)
        self.create_subscription(
            CustomMsg, '/livox/lidar',
            lambda message: self._sensor_callback('livox_lidar', message), qos)
        self.create_subscription(
            Imu, '/livox/imu',
            lambda message: self._sensor_callback('livox_imu', message), qos)
        self.create_subscription(
            PathMessage, '/lio/path', self._path_callback, qos)
        self.create_subscription(
            PointCloud2, '/lio/cloud_world', self._cloud_callback, qos)
        self._heartbeat = self.create_timer(5.0, self._log_progress)
        self.get_logger().info(f'Evaluation output: {self._output_dir}')

    def _sensor_callback(self, name, message):
        self._total_counts[name] += 1
        self._sensor_records[name].append(
            (stamp_seconds(message.header.stamp), time.monotonic()))

    def _path_callback(self, message):
        self._total_counts['lio_path'] += 1
        self._sensor_records['lio_path'].append(
            (stamp_seconds(message.header.stamp), time.monotonic()))

    def _pose_callback(self, name, message):
        self._total_counts[name] += 1
        if not finite_odometry(message):
            self._nonfinite_counts[name] += 1
            return
        pose = message.pose.pose
        quaternion = pose.orientation
        self._pose_records[name].append({
            'stamp': stamp_seconds(message.header.stamp),
            'receipt': time.monotonic(),
            'frame_id': message.header.frame_id,
            'child_frame_id': message.child_frame_id,
            'x': pose.position.x,
            'y': pose.position.y,
            'z': pose.position.z,
            'qx': quaternion.x,
            'qy': quaternion.y,
            'qz': quaternion.z,
            'qw': quaternion.w,
            'yaw': quaternion_yaw(quaternion),
        })

    def _cloud_callback(self, message):
        self._total_counts['lio_cloud_world'] += 1
        if self._map_tree is None:
            return
        self._cloud_callbacks += 1
        if self._cloud_callbacks % self._cloud_stride != 0:
            return
        try:
            values = point_cloud2.read_points(
                message, field_names=('x', 'y', 'z'), skip_nans=True)
            if len(values) == 0:
                return
            if isinstance(values, np.ndarray) and values.dtype.names:
                xyz = np.column_stack(
                    (values['x'], values['y'], values['z']))
            else:
                xyz = np.asarray(list(values), dtype=np.float64)
            stride = max(1, len(xyz) // 400)
            sample = np.asarray(xyz[::stride][:400], dtype=np.float64)
            if sample.ndim != 2 or sample.shape[1] != 3:
                return
            distances, _ = self._map_tree.query(sample, k=1, workers=1)
            self._cloud_distances.extend(distances.tolist())
        except Exception as error:  # Keep localization evaluation alive.
            self.get_logger().warning(f'Cloud overlap sample failed: {error}')

    def _log_progress(self):
        counts = ', '.join(
            f'{name}={count}' for name, count in sorted(self._total_counts.items()))
        self.get_logger().info(f'capture progress: {counts}')

    @staticmethod
    def _array_stats(values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return None
        return {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'rmse': float(np.sqrt(np.mean(np.square(values)))),
            'max': float(np.max(values)),
            'p95': float(np.percentile(values, 95.0)),
        }

    def _topic_summary(self, name, records):
        if not records:
            return {
                'count': self._total_counts[name],
                'valid_count': 0,
                'nonfinite_count': self._nonfinite_counts[name],
            }
        if isinstance(records[0], dict):
            stamps = np.asarray([record['stamp'] for record in records])
            receipts = np.asarray([record['receipt'] for record in records])
            quaternion_norms = np.asarray([
                math.sqrt(record['qx'] ** 2 + record['qy'] ** 2 +
                          record['qz'] ** 2 + record['qw'] ** 2)
                for record in records
            ])
        else:
            stamps = np.asarray([record[0] for record in records])
            receipts = np.asarray([record[1] for record in records])
            quaternion_norms = None
        stamp_deltas = np.diff(stamps)
        receipt_deltas = np.diff(receipts)
        stamp_span = float(stamps[-1] - stamps[0])
        receipt_span = float(receipts[-1] - receipts[0])
        expected_receipt = (stamps - stamps[0]) / self._playback_rate
        scheduling_latency = (receipts - receipts[0]) - expected_receipt
        positive = stamp_deltas[stamp_deltas > 0.0]
        median_delta = float(np.median(positive)) if positive.size else math.inf
        stale_threshold = max(0.08, 3.0 * median_delta)
        summary = {
            'count': self._total_counts[name],
            'valid_count': len(records),
            'nonfinite_count': self._nonfinite_counts[name],
            'stamp_span_sec': stamp_span,
            'receipt_span_sec': receipt_span,
            'header_rate_hz': ((len(records) - 1) / stamp_span
                               if stamp_span > 0.0 else 0.0),
            'receipt_rate_hz': ((len(records) - 1) / receipt_span
                                if receipt_span > 0.0 else 0.0),
            'timestamp_nonmonotonic_count': int(np.sum(stamp_deltas <= 0.0)),
            'stale_gap_threshold_sec': stale_threshold,
            'stale_gap_count': int(np.sum(stamp_deltas > stale_threshold)),
            'max_stamp_gap_sec': float(np.max(stamp_deltas)) if stamp_deltas.size else 0.0,
            'relative_scheduling_latency_sec': self._array_stats(
                np.abs(scheduling_latency)),
        }
        if quaternion_norms is not None:
            summary['quaternion_norm_min'] = float(np.min(quaternion_norms))
            summary['quaternion_norm_max'] = float(np.max(quaternion_norms))
            distances = []
            yaw_jumps = []
            for previous, current in zip(records[:-1], records[1:]):
                distances.append(math.sqrt(
                    (current['x'] - previous['x']) ** 2 +
                    (current['y'] - previous['y']) ** 2 +
                    (current['z'] - previous['z']) ** 2))
                yaw_jumps.append(abs(wrapped_angle(
                    current['yaw'] - previous['yaw'])))
            summary['position_jump_over_1m_count'] = int(
                np.sum(np.asarray(distances) > 1.0))
            summary['max_position_step_m'] = max(distances, default=0.0)
            summary['yaw_jump_over_45deg_count'] = int(
                np.sum(np.asarray(yaw_jumps) > math.radians(45.0)))
            summary['max_yaw_step_deg'] = math.degrees(max(yaw_jumps, default=0.0))
            summary['frames'] = sorted(set(record['frame_id'] for record in records))
            summary['child_frames'] = sorted(
                set(record['child_frame_id'] for record in records))
        return summary

    def _compare_with_reference(self, source_name, reference):
        source = self._pose_records[source_name]
        if not source or not reference:
            return [], None
        reference_stamps = [record['stamp'] for record in reference]
        rows = []
        for record in source:
            index = bisect.bisect_left(reference_stamps, record['stamp'])
            choices = [candidate for candidate in (index - 1, index)
                       if 0 <= candidate < len(reference)]
            if not choices:
                continue
            chosen = min(choices, key=lambda candidate: abs(
                reference[candidate]['stamp'] - record['stamp']))
            target = reference[chosen]
            time_error = target['stamp'] - record['stamp']
            if abs(time_error) > 0.05:
                continue
            dx = record['x'] - target['x']
            dy = record['y'] - target['y']
            dz = record['z'] - target['z']
            planar = math.hypot(dx, dy)
            error_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
            yaw_error = abs(wrapped_angle(record['yaw'] - target['yaw']))
            rows.append({
                'source_stamp': record['stamp'],
                'reference_stamp': target['stamp'],
                'time_error_sec': time_error,
                'dx_m': dx, 'dy_m': dy, 'dz_m': dz,
                'planar_error_m': planar,
                'position_3d_error_m': error_3d,
                'yaw_error_deg': math.degrees(yaw_error),
            })
        if not rows:
            return rows, None
        summary = {
            'alignment': 'direct; bag TF map->odom is identity and both poses are IMU-frame poses',
            'reference_is_ground_truth': False,
            'matched_count': len(rows),
            'planar_error_m': self._array_stats(
                [row['planar_error_m'] for row in rows]),
            'position_3d_error_m': self._array_stats(
                [row['position_3d_error_m'] for row in rows]),
            'yaw_error_deg': self._array_stats(
                [row['yaw_error_deg'] for row in rows]),
            'time_alignment_abs_sec': self._array_stats(
                [abs(row['time_error_sec']) for row in rows]),
        }
        return rows, summary

    @staticmethod
    def _write_csv(path, rows):
        if not rows:
            return
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def write_results(self):
        if self._written:
            return
        self._written = True
        self._output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            'evaluation_wall_duration_sec': time.monotonic() - self._start_monotonic,
            'playback_rate': self._playback_rate,
            'topics': {},
            'comparisons': {},
        }
        all_names = set(self._sensor_records) | set(self._pose_records) | set(self._total_counts)
        for name in sorted(all_names):
            records = self._pose_records.get(name) or self._sensor_records.get(name) or []
            summary['topics'][name] = self._topic_summary(name, records)
            if name in self._pose_records:
                self._write_csv(self._output_dir / f'{name}.csv', records)

        lidar = self._sensor_records.get('livox_lidar', [])
        if lidar:
            first_lidar_stamp = lidar[0][0]
            for name in ('lio_odom', 'lio_robo_odom', 'lio_imu_odom', 'policy_odom_50hz'):
                if self._pose_records.get(name):
                    summary['topics'][name]['initialization_latency_stamp_sec'] = (
                        self._pose_records[name][0]['stamp'] - first_lidar_stamp)

        reference = self._pose_records.get('reference_glim_odom', [])
        for source_name in ('lio_odom', 'lio_robo_odom'):
            rows, comparison = self._compare_with_reference(source_name, reference)
            if comparison is not None:
                summary['comparisons'][source_name] = comparison
                self._write_csv(
                    self._output_dir / f'{source_name}_vs_glim_reference.csv', rows)

        if self._cloud_distances:
            distances = np.asarray(self._cloud_distances)
            summary['cloud_map_overlap'] = {
                'sample_count': int(distances.size),
                'nearest_map_distance_m': self._array_stats(distances),
                'fraction_within_0_25m': float(np.mean(distances <= 0.25)),
                'fraction_within_0_50m': float(np.mean(distances <= 0.50)),
                'fraction_within_1_00m': float(np.mean(distances <= 1.00)),
            }
        else:
            summary['cloud_map_overlap'] = {'sample_count': 0}

        with (self._output_dir / 'summary.json').open('w') as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
        with (self._output_dir / 'summary.txt').open('w') as stream:
            stream.write(json.dumps(summary, indent=2, sort_keys=True))
            stream.write('\n')
        self.get_logger().info(f'Wrote evaluation results to {self._output_dir}')


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationEvaluator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RuntimeError):
        pass
    finally:
        node.write_results()
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, RuntimeError):
            pass

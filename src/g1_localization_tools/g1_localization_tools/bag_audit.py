import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from .common import quaternion_yaw, stamp_seconds


SELECTED_TOPICS = {
    '/tf', '/tf_static', '/livox/lidar', '/livox/imu',
    '/glim_ros/odom_corrected', '/odometry/odom',
}


def find_stationary_windows(records):
    if len(records) < 2:
        return []
    stamps = np.asarray([record['stamp'] for record in records])
    positions = np.asarray([record['position'] for record in records])
    yaws = np.unwrap(np.asarray([record['yaw'] for record in records]))
    delta_time = np.diff(stamps)
    planar_speed = np.linalg.norm(
        np.diff(positions[:, :2], axis=0), axis=1) / delta_time
    yaw_rate = np.abs(np.diff(yaws)) / delta_time
    good = ((planar_speed < 0.05) & (yaw_rate < 0.08) &
            (delta_time > 0.0) & (delta_time < 0.2))
    windows = []
    start = None
    for index, is_good in enumerate(good):
        if is_good and start is None:
            start = index
        end_run = start is not None and (not is_good or index == len(good) - 1)
        if end_run:
            end = index + 1 if is_good and index == len(good) - 1 else index
            duration = stamps[end] - stamps[start]
            if duration >= 1.0:
                windows.append({
                    'start_offset_sec': float(stamps[start] - stamps[0]),
                    'end_offset_sec': float(stamps[end] - stamps[0]),
                    'duration_sec': float(duration),
                    'max_planar_speed_mps': float(np.max(planar_speed[start:end])),
                    'max_yaw_rate_radps': float(np.max(yaw_rate[start:end])),
                })
            start = None
    return windows


def audit_bag(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', ''),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    classes = {name: get_message(topic_types[name])
               for name in SELECTED_TOPICS if name in topic_types}
    counts = Counter()
    record_times = defaultdict(list)
    frames = defaultdict(Counter)
    transforms = {}
    odometry = defaultdict(list)
    lidar_point_counts = []
    lidar_offsets = []

    while reader.has_next():
        topic, serialized, record_nanoseconds = reader.read_next()
        if topic not in classes:
            continue
        message = deserialize_message(serialized, classes[topic])
        counts[topic] += 1
        record_times[topic].append(record_nanoseconds * 1.0e-9)
        if topic in ('/tf', '/tf_static'):
            for transform in message.transforms:
                key = f'{transform.header.frame_id}->{transform.child_frame_id}'
                if (topic, key) not in transforms:
                    translation = transform.transform.translation
                    rotation = transform.transform.rotation
                    transforms[(topic, key)] = {
                        'topic': topic,
                        'parent': transform.header.frame_id,
                        'child': transform.child_frame_id,
                        'translation': [translation.x, translation.y, translation.z],
                        'quaternion_xyzw': [
                            rotation.x, rotation.y, rotation.z, rotation.w],
                    }
        elif topic in ('/glim_ros/odom_corrected', '/odometry/odom'):
            pose = message.pose.pose
            frames[topic][(message.header.frame_id, message.child_frame_id)] += 1
            odometry[topic].append({
                'stamp': stamp_seconds(message.header.stamp),
                'position': [pose.position.x, pose.position.y, pose.position.z],
                'quaternion_xyzw': [
                    pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w],
                'yaw': quaternion_yaw(pose.orientation),
            })
        elif topic == '/livox/imu':
            frames[topic][message.header.frame_id] += 1
        elif topic == '/livox/lidar':
            frames[topic][message.header.frame_id] += 1
            lidar_point_counts.append(message.point_num)
            lidar_offsets.append(message.points[-1].offset_time if message.points else 0)

    summary = {'bag_path': str(bag_path), 'topics': {}, 'frames': {}, 'transforms': []}
    for topic, topic_type in sorted(topic_types.items()):
        times = record_times.get(topic, [])
        entry = {'type': topic_type, 'selected_message_count': counts[topic]}
        if len(times) > 1:
            span = times[-1] - times[0]
            entry.update({'record_span_sec': span,
                          'average_record_rate_hz': (len(times) - 1) / span})
        summary['topics'][topic] = entry
    for topic, values in frames.items():
        summary['frames'][topic] = [
            {'frame': str(frame), 'count': count} for frame, count in values.items()]
    summary['transforms'] = [transforms[key] for key in sorted(transforms)]
    if lidar_point_counts:
        summary['lidar'] = {
            'point_count_min': min(lidar_point_counts),
            'point_count_mean': float(np.mean(lidar_point_counts)),
            'point_count_max': max(lidar_point_counts),
            'last_offset_time_ns_min': min(lidar_offsets),
            'last_offset_time_ns_mean': float(np.mean(lidar_offsets)),
            'last_offset_time_ns_max': max(lidar_offsets),
        }
    if odometry.get('/odometry/odom'):
        records = odometry['/odometry/odom']
        summary['initial_map_lidar_pose'] = records[0]
        summary['stationary_windows'] = find_stationary_windows(records)
    if odometry.get('/glim_ros/odom_corrected'):
        summary['initial_map_imu_reference_pose'] = (
            odometry['/glim_ros/odom_corrected'][0])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit_bag(Path(args.bag))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f'Wrote bag audit: {output}')


if __name__ == '__main__':
    main()

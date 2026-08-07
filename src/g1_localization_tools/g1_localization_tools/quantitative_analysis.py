import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import rosbag2_py
from livox_ros_driver2.msg import CustomMsg
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message

from .trajectory_overlay import (
    interpolate_sample,
    normalized_quaternion,
    pose_matrix,
    quaternion_conjugate,
    quaternion_multiply,
    sample_signature,
)


ODOMETRY_TYPE = 'nav_msgs/msg/Odometry'
LIVOX_TYPE = 'livox_ros_driver2/msg/CustomMsg'
SUPER_LABEL = 'Super-LIO'
ODOM_GT_COMPARISON = 'Super-LIO vs GLIM odom_corrected'
IMU_GT_COMPARISON = 'Super-LIO vs GLIM traj_imu'


def stamp_nanoseconds(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def read_bag(bag_path, odometry_topics=(), read_lidar=False):
    odometry_topics = set(odometry_topics)
    odometry = {topic: [] for topic in odometry_topics}
    lidar = []
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', ''),
    )
    types = {
        item.name: item.type for item in reader.get_all_topics_and_types()}
    for topic in odometry_topics:
        if types.get(topic) != ODOMETRY_TYPE:
            raise RuntimeError(f'{bag_path}: missing {ODOMETRY_TYPE} {topic}')
    if read_lidar and types.get('/livox/lidar') != LIVOX_TYPE:
        raise RuntimeError(f'{bag_path}: missing {LIVOX_TYPE} /livox/lidar')

    while reader.has_next():
        topic, serialized, record_ns = reader.read_next()
        if topic in odometry_topics:
            message = deserialize_message(serialized, Odometry)
            pose = message.pose.pose
            odometry[topic].append({
                'stamp_ns': stamp_nanoseconds(message.header.stamp),
                'record_ns': int(record_ns),
                'position': np.asarray([
                    pose.position.x, pose.position.y, pose.position.z],
                    dtype=np.float64),
                'quaternion': np.asarray([
                    pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w],
                    dtype=np.float64),
            })
        elif read_lidar and topic == '/livox/lidar':
            message = deserialize_message(serialized, CustomMsg)
            start_ns = stamp_nanoseconds(message.header.stamp)
            end_ns = start_ns
            if message.points:
                end_ns += int(message.points[-1].offset_time)
            lidar.append({
                'start_ns': start_ns,
                'end_ns': end_ns,
                'record_ns': int(record_ns),
            })

    for records in odometry.values():
        records.sort(key=lambda item: item['stamp_ns'])
    lidar.sort(key=lambda item: item['start_ns'])
    return odometry, lidar


def read_text_trajectory(path):
    """Read `timestamp x y z qx qy qz qw` GLIM trajectory text."""
    records = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            fields = stripped.split()
            if len(fields) < 8:
                raise RuntimeError(
                    f'{path}:{line_number}: expected at least 8 columns')
            values = [float(field) for field in fields[:8]]
            records.append({
                'stamp_ns': int(round(values[0] * 1.0e9)),
                'position': np.asarray(values[1:4], dtype=np.float64),
                'quaternion': normalized_quaternion(np.asarray(
                    values[4:8], dtype=np.float64)),
            })
    records.sort(key=lambda item: item['stamp_ns'])
    if len(records) < 2:
        raise RuntimeError(f'{path}: fewer than two trajectory poses')
    return records


def quaternion_yaw(quaternion):
    x, y, z, w = normalized_quaternion(quaternion)
    return math.atan2(
        2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))


def wrapped_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def orientation_error_degrees(reference, estimate):
    relative = normalized_quaternion(quaternion_multiply(
        quaternion_conjugate(reference), estimate))
    return math.degrees(2.0 * math.acos(
        max(-1.0, min(1.0, abs(float(relative[3]))))))


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(values) < 3:
        return values.copy()
    window = min(int(window), len(values))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode='edge')
    kernel = np.ones(window, dtype=np.float64) / window
    return np.column_stack([
        np.convolve(padded[:, axis], kernel, mode='valid')
        for axis in range(values.shape[1])
    ])


def rolling_rmse(values, window):
    squared = np.asarray(values, dtype=np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    output = np.empty(len(squared), dtype=np.float64)
    for index in range(len(squared)):
        start = max(0, index + 1 - window)
        output[index] = math.sqrt(
            (cumulative[index + 1] - cumulative[start]) /
            (index + 1 - start))
    return output


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {'count': 0}
    return {
        'count': int(len(values)),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'rmse': float(math.sqrt(np.mean(values ** 2))),
        'p95': float(np.percentile(values, 95)),
        'p99': float(np.percentile(values, 99)),
        'max': float(np.max(values)),
    }


def safe_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if len(left) < 3 or np.std(left) < 1.0e-12 or np.std(right) < 1.0e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def rank_values(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def binned_statistics(values, errors, label, algorithm):
    values = np.asarray(values, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    edges = np.unique(np.percentile(values, [0, 25, 50, 75, 100]))
    if len(edges) < 2:
        return []
    rows = []
    for index in range(len(edges) - 1):
        lower = edges[index]
        upper = edges[index + 1]
        if index == len(edges) - 2:
            mask = (values >= lower) & (values <= upper)
        else:
            mask = (values >= lower) & (values < upper)
        current = errors[mask]
        if len(current) == 0:
            continue
        rows.append({
            'algorithm': algorithm,
            'metric': label,
            'bin_index': index,
            'lower': float(lower),
            'upper': float(upper),
            'count': int(len(current)),
            'rmse_m': float(math.sqrt(np.mean(current ** 2))),
            'mean_m': float(np.mean(current)),
            'p95_m': float(np.percentile(current, 95)),
            'max_m': float(np.max(current)),
        })
    return rows


def match_latency(outputs, lidar, lidar_stamp_field):
    lidar_stamps = np.asarray(
        [record[lidar_stamp_field] for record in lidar], dtype=np.int64)
    rows = []
    for output in outputs:
        stamp_ns = output['stamp_ns']
        index = int(np.searchsorted(lidar_stamps, stamp_ns))
        candidates = [item for item in (index - 1, index)
                      if 0 <= item < len(lidar)]
        if not candidates:
            continue
        matched = min(
            candidates,
            key=lambda item: abs(int(lidar_stamps[item]) - stamp_ns))
        input_record = lidar[matched]
        rows.append({
            'stamp_ns': stamp_ns,
            'record_ns': output['record_ns'],
            'latency_ms': (
                output['record_ns'] - input_record['record_ns']) * 1.0e-6,
            'sensor_stamp_residual_ms': (
                stamp_ns - int(lidar_stamps[matched])) * 1.0e-6,
        })
    return rows


def contiguous_episodes(mask, times, values):
    episodes = []
    start = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        at_end = start is not None and (not active or index == len(mask) - 1)
        if at_end:
            end = index if active and index == len(mask) - 1 else index - 1
            episodes.append((
                start, end, float(times[start]), float(times[end]),
                float(np.max(values[start:end + 1]))))
            start = None
    return episodes


def write_csv(path, fieldnames, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_correlation(value):
    return 'N/A' if value is None else f'{value:+.3f}'


def analyze(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    map_data, _ = read_bag(
        args.map_frame_bag,
        odometry_topics=(args.map_frame_topic, args.replayed_input_topic))
    reference_data, reference_lidar = read_bag(
        args.reference_bag,
        odometry_topics=(args.reference_topic,
                         args.calibration_reference_topic),
        read_lidar=True)
    estimate_data, estimate_lidar = read_bag(
        args.estimated_bag,
        odometry_topics=(args.estimated_topic,), read_lidar=True)
    imu_trajectory = read_text_trajectory(args.imu_trajectory)

    replayed_signatures = {
        sample_signature(
            record['stamp_ns'], record['position'], record['quaternion'])
        for record in map_data[args.replayed_input_topic]
    }
    all_estimates = estimate_data[args.estimated_topic]
    estimates = [
        record for record in all_estimates
        if sample_signature(
            record['stamp_ns'], record['position'], record['quaternion'])
        not in replayed_signatures
    ]
    filtered_count = len(all_estimates) - len(estimates)
    if len(estimates) < 2:
        raise RuntimeError('fewer than two localization-only estimates')

    calibration_reference = reference_data[
        args.calibration_reference_topic][0]
    map_anchor = interpolate_sample(
        map_data[args.map_frame_topic],
        calibration_reference['stamp_ns'], max_gap_ns=100_000_000)
    if map_anchor is None:
        raise RuntimeError('failed to interpolate map-frame calibration pose')
    map_from_glim = (
        pose_matrix(map_anchor) @
        np.linalg.inv(pose_matrix(calibration_reference)))
    map_from_glim_quaternion = normalized_quaternion(quaternion_multiply(
        map_anchor['quaternion'],
        quaternion_conjugate(calibration_reference['quaternion'])))

    reference = reference_data[args.reference_topic]
    matched = []
    for estimate in estimates:
        ground_truth = interpolate_sample(
            reference, estimate['stamp_ns'], max_gap_ns=100_000_000)
        imu_estimate = interpolate_sample(
            imu_trajectory, estimate['stamp_ns'], max_gap_ns=100_000_000)
        if ground_truth is None or imu_estimate is None:
            continue
        reference_homogeneous = np.append(ground_truth['position'], 1.0)
        ground_truth_position = (
            map_from_glim @ reference_homogeneous)[:3]
        ground_truth_quaternion = normalized_quaternion(quaternion_multiply(
            map_from_glim_quaternion, ground_truth['quaternion']))
        imu_position = (
            map_from_glim @ np.append(imu_estimate['position'], 1.0))[:3]
        imu_quaternion = normalized_quaternion(quaternion_multiply(
            map_from_glim_quaternion, imu_estimate['quaternion']))
        matched.append({
            'estimate': estimate,
            'ground_truth_position': ground_truth_position,
            'ground_truth_quaternion': ground_truth_quaternion,
            'imu_position': imu_position,
            'imu_quaternion': imu_quaternion,
        })
    if len(matched) < 2:
        raise RuntimeError('fewer than two time-matched poses')

    stamps_ns = np.asarray(
        [item['estimate']['stamp_ns'] for item in matched], dtype=np.int64)
    times = (stamps_ns - stamps_ns[0]) * 1.0e-9
    estimate_positions = np.asarray(
        [item['estimate']['position'] for item in matched])
    estimate_quaternions = np.asarray(
        [item['estimate']['quaternion'] for item in matched])
    imu_positions = np.asarray([item['imu_position'] for item in matched])
    imu_quaternions = np.asarray([
        item['imu_quaternion'] for item in matched])
    ground_truth_positions = np.asarray(
        [item['ground_truth_position'] for item in matched])
    ground_truth_quaternions = np.asarray([
        item['ground_truth_quaternion'] for item in matched])
    error_vectors = estimate_positions - ground_truth_positions
    position_errors = np.linalg.norm(error_vectors, axis=1)
    imu_error_vectors = estimate_positions - imu_positions
    imu_position_errors = np.linalg.norm(imu_error_vectors, axis=1)

    orientation_errors = np.asarray([
        orientation_error_degrees(
            item['ground_truth_quaternion'], item['estimate']['quaternion'])
        for item in matched
    ])
    yaw_errors = np.asarray([
        math.degrees(wrapped_angle(
            quaternion_yaw(item['estimate']['quaternion']) -
            quaternion_yaw(item['ground_truth_quaternion'])))
        for item in matched
    ])
    imu_orientation_errors = np.asarray([
        orientation_error_degrees(imu_quaternion, estimate_quaternion)
        for imu_quaternion, estimate_quaternion in zip(
            imu_quaternions, estimate_quaternions)
    ])
    imu_yaw_errors = np.asarray([
        math.degrees(wrapped_angle(
            quaternion_yaw(estimate_quaternion) -
            quaternion_yaw(imu_quaternion)))
        for imu_quaternion, estimate_quaternion in zip(
            imu_quaternions, estimate_quaternions)
    ])

    smoothed_ground_truth = moving_average(
        ground_truth_positions, args.motion_smoothing_samples)
    velocity_vectors = np.gradient(smoothed_ground_truth, times, axis=0)
    acceleration_vectors = np.gradient(velocity_vectors, times, axis=0)
    speeds = np.linalg.norm(velocity_vectors, axis=1)
    accelerations = np.linalg.norm(acceleration_vectors, axis=1)
    smoothed_imu_ground_truth = moving_average(
        imu_positions, args.motion_smoothing_samples)
    imu_ground_truth_velocity = np.gradient(
        smoothed_imu_ground_truth, times, axis=0)
    imu_ground_truth_acceleration = np.gradient(
        imu_ground_truth_velocity, times, axis=0)
    imu_speeds = np.linalg.norm(imu_ground_truth_velocity, axis=1)
    imu_accelerations = np.linalg.norm(
        imu_ground_truth_acceleration, axis=1)

    estimate_velocity = np.gradient(
        moving_average(estimate_positions, args.motion_smoothing_samples),
        times, axis=0)
    velocity_residual = np.linalg.norm(
        estimate_velocity - velocity_vectors, axis=1)
    imu_velocity_residual = np.linalg.norm(
        estimate_velocity - imu_ground_truth_velocity, axis=1)
    jump_steps = np.concatenate((
        [0.0], np.linalg.norm(np.diff(error_vectors, axis=0), axis=1)))
    imu_jump_steps = np.concatenate((
        [0.0], np.linalg.norm(np.diff(imu_error_vectors, axis=0), axis=1)))
    orientation_steps = np.concatenate((
        [0.0], np.abs(np.diff(orientation_errors))))
    imu_orientation_steps = np.concatenate((
        [0.0], np.abs(np.diff(imu_orientation_errors))))
    cumulative_rmse = np.sqrt(
        np.cumsum(position_errors ** 2) /
        np.arange(1, len(position_errors) + 1))
    imu_cumulative_rmse = np.sqrt(
        np.cumsum(imu_position_errors ** 2) /
        np.arange(1, len(imu_position_errors) + 1))
    median_dt = float(np.median(np.diff(times)))
    rolling_window = max(1, int(round(args.rolling_window_sec / median_dt)))
    rolling_position_rmse = rolling_rmse(position_errors, rolling_window)
    imu_rolling_position_rmse = rolling_rmse(
        imu_position_errors, rolling_window)

    reference_latency_rows = match_latency(
        reference, reference_lidar, 'start_ns')
    estimate_latency_rows = match_latency(
        estimates, estimate_lidar, 'end_ns')
    reference_latency = np.asarray(
        [row['latency_ms'] for row in reference_latency_rows])
    estimate_latency = np.asarray(
        [row['latency_ms'] for row in estimate_latency_rows])
    estimate_latency_start = estimate_latency_rows[0]['stamp_ns']
    estimate_latency_times = np.asarray([
        (row['stamp_ns'] - estimate_latency_start) * 1.0e-9
        for row in estimate_latency_rows
    ])
    steady_latency_mask = (
        estimate_latency_times >= args.latency_steady_state_start_sec)
    estimate_steady_latency = estimate_latency[steady_latency_mask]

    speed_pearson = safe_correlation(speeds, position_errors)
    acceleration_pearson = safe_correlation(accelerations, position_errors)
    speed_spearman = safe_correlation(
        rank_values(speeds), rank_values(position_errors))
    acceleration_spearman = safe_correlation(
        rank_values(accelerations), rank_values(position_errors))
    imu_speed_pearson = safe_correlation(imu_speeds, imu_position_errors)
    imu_acceleration_pearson = safe_correlation(
        imu_accelerations, imu_position_errors)
    imu_speed_spearman = safe_correlation(
        rank_values(imu_speeds), rank_values(imu_position_errors))
    imu_acceleration_spearman = safe_correlation(
        rank_values(imu_accelerations), rank_values(imu_position_errors))

    jump_mask = jump_steps > args.jump_threshold_m
    orientation_jump_mask = (
        orientation_steps > args.orientation_jump_threshold_deg)
    error_mask = position_errors > args.position_error_threshold_m
    imu_jump_mask = imu_jump_steps > args.jump_threshold_m
    imu_orientation_jump_mask = (
        imu_orientation_steps > args.orientation_jump_threshold_deg)
    imu_error_mask = imu_position_errors > args.position_error_threshold_m
    latency_spike_mask = estimate_latency > args.latency_spike_threshold_ms
    steady_latency_spike_mask = latency_spike_mask & steady_latency_mask

    events = []
    for index in np.flatnonzero(jump_mask):
        events.append({
            'algorithm': ODOM_GT_COMPARISON,
            'type': 'position_jump',
            'start_sec': float(times[index]),
            'end_sec': float(times[index]),
            'value': float(jump_steps[index]),
            'unit': 'm/frame',
        })
    for index in np.flatnonzero(orientation_jump_mask):
        events.append({
            'algorithm': ODOM_GT_COMPARISON,
            'type': 'orientation_jump',
            'start_sec': float(times[index]),
            'end_sec': float(times[index]),
            'value': float(orientation_steps[index]),
            'unit': 'deg/frame',
        })
    for start, end, start_sec, end_sec, maximum in contiguous_episodes(
            error_mask, times, position_errors):
        events.append({
            'algorithm': ODOM_GT_COMPARISON,
            'type': 'position_error_episode',
            'start_sec': start_sec,
            'end_sec': end_sec,
            'value': maximum,
            'unit': 'm',
        })
    for index in np.flatnonzero(imu_jump_mask):
        events.append({
            'algorithm': IMU_GT_COMPARISON,
            'type': 'position_jump',
            'start_sec': float(times[index]),
            'end_sec': float(times[index]),
            'value': float(imu_jump_steps[index]),
            'unit': 'm/frame',
        })
    for index in np.flatnonzero(imu_orientation_jump_mask):
        events.append({
            'algorithm': IMU_GT_COMPARISON,
            'type': 'orientation_jump',
            'start_sec': float(times[index]),
            'end_sec': float(times[index]),
            'value': float(imu_orientation_steps[index]),
            'unit': 'deg/frame',
        })
    for start, end, start_sec, end_sec, maximum in contiguous_episodes(
            imu_error_mask, times, imu_position_errors):
        events.append({
            'algorithm': IMU_GT_COMPARISON,
            'type': 'position_error_episode',
            'start_sec': start_sec,
            'end_sec': end_sec,
            'value': maximum,
            'unit': 'm',
        })
    for index in np.flatnonzero(latency_spike_mask):
        event_time = (
            estimate_latency_rows[index]['stamp_ns'] -
            estimate_latency_start) * 1.0e-9
        events.append({
            'algorithm': SUPER_LABEL,
            'type': 'latency_spike',
            'start_sec': float(event_time),
            'end_sec': float(event_time),
            'value': float(estimate_latency[index]),
            'unit': 'ms',
        })
    events.sort(key=lambda event: (event['start_sec'], event['type']))

    motion_bins = (
        binned_statistics(
            speeds, position_errors, 'speed_mps', ODOM_GT_COMPARISON) +
        binned_statistics(
            accelerations, position_errors, 'acceleration_mps2',
            ODOM_GT_COMPARISON) +
        binned_statistics(
            imu_speeds, imu_position_errors, 'speed_mps',
            IMU_GT_COMPARISON) +
        binned_statistics(
            imu_accelerations, imu_position_errors, 'acceleration_mps2',
            IMU_GT_COMPARISON))

    sample_rows = []
    for index in range(len(times)):
        sample_rows.append({
            'time_sec': float(times[index]),
            'sensor_stamp_ns': int(stamps_ns[index]),
            'gt_x_m': float(ground_truth_positions[index, 0]),
            'gt_y_m': float(ground_truth_positions[index, 1]),
            'gt_z_m': float(ground_truth_positions[index, 2]),
            'super_x_m': float(estimate_positions[index, 0]),
            'super_y_m': float(estimate_positions[index, 1]),
            'super_z_m': float(estimate_positions[index, 2]),
            'imu_x_m': float(imu_positions[index, 0]),
            'imu_y_m': float(imu_positions[index, 1]),
            'imu_z_m': float(imu_positions[index, 2]),
            'error_x_m': float(error_vectors[index, 0]),
            'error_y_m': float(error_vectors[index, 1]),
            'error_z_m': float(error_vectors[index, 2]),
            'position_error_m': float(position_errors[index]),
            'cumulative_rmse_m': float(cumulative_rmse[index]),
            'rolling_rmse_m': float(rolling_position_rmse[index]),
            'orientation_error_deg': float(orientation_errors[index]),
            'yaw_error_deg': float(yaw_errors[index]),
            'speed_mps': float(speeds[index]),
            'acceleration_mps2': float(accelerations[index]),
            'imu_speed_mps': float(imu_speeds[index]),
            'imu_acceleration_mps2': float(imu_accelerations[index]),
            'velocity_residual_mps': float(velocity_residual[index]),
            'position_jump_m': float(jump_steps[index]),
            'orientation_jump_deg': float(orientation_steps[index]),
            'imu_error_x_m': float(imu_error_vectors[index, 0]),
            'imu_error_y_m': float(imu_error_vectors[index, 1]),
            'imu_error_z_m': float(imu_error_vectors[index, 2]),
            'imu_position_error_m': float(imu_position_errors[index]),
            'imu_cumulative_rmse_m': float(imu_cumulative_rmse[index]),
            'imu_rolling_rmse_m': float(imu_rolling_position_rmse[index]),
            'imu_orientation_error_deg': float(
                imu_orientation_errors[index]),
            'imu_yaw_error_deg': float(imu_yaw_errors[index]),
            'imu_velocity_residual_mps': float(
                imu_velocity_residual[index]),
            'imu_position_jump_m': float(imu_jump_steps[index]),
            'imu_orientation_jump_deg': float(
                imu_orientation_steps[index]),
        })
    sample_fields = list(sample_rows[0])
    write_csv(output_dir / 'timeseries.csv', sample_fields, sample_rows)

    latency_rows = []
    for algorithm, rows in (
            ('GLIM', reference_latency_rows),
            ('Super-LIO', estimate_latency_rows)):
        origin = rows[0]['stamp_ns']
        for row in rows:
            latency_rows.append({
                'algorithm': algorithm,
                'time_sec': (row['stamp_ns'] - origin) * 1.0e-9,
                'latency_ms': float(row['latency_ms']),
                'sensor_stamp_residual_ms': float(
                    row['sensor_stamp_residual_ms']),
            })
    write_csv(
        output_dir / 'latency.csv',
        ['algorithm', 'time_sec', 'latency_ms',
         'sensor_stamp_residual_ms'], latency_rows)
    write_csv(
        output_dir / 'events.csv',
        ['algorithm', 'type', 'start_sec', 'end_sec', 'value', 'unit'],
        events)
    write_csv(
        output_dir / 'motion_bins.csv',
        ['algorithm', 'metric', 'bin_index', 'lower', 'upper', 'count',
         'rmse_m', 'mean_m', 'p95_m', 'max_m'], motion_bins)

    position_summary = percentile_summary(position_errors)
    orientation_summary = percentile_summary(orientation_errors)
    jump_summary = percentile_summary(jump_steps[1:])
    velocity_residual_summary = percentile_summary(velocity_residual)
    imu_position_summary = percentile_summary(imu_position_errors)
    imu_orientation_summary = percentile_summary(imu_orientation_errors)
    imu_jump_summary = percentile_summary(imu_jump_steps[1:])
    imu_velocity_residual_summary = percentile_summary(
        imu_velocity_residual)
    glim_latency_summary = percentile_summary(reference_latency)
    super_latency_summary = percentile_summary(estimate_latency)
    super_steady_latency_summary = percentile_summary(estimate_steady_latency)
    summary = {
        'method': {
            'ground_truth_candidates': [
                'GLIM odom_corrected', 'GLIM traj_imu'],
            'odom_reference_source': (
                f'{Path(args.reference_bag).resolve()}:'
                f'{args.reference_topic}'),
            'imu_reference_source': str(
                Path(args.imu_trajectory).resolve()),
            'spatial_alignment': 'one fixed map<-GLIM transform from one '
                                 'independent common sensor-time pose; '
                                 'applied equally to both GLIM references',
            'time_alignment': 'raw sensor timestamps for all trajectories; '
                              'no fitted offset',
            'imu_trajectory_format': 'timestamp x y z qx qy qz qw',
            'latency_definition': 'bag record time(output) - bag record '
                                  'time(matched LiDAR scan)',
            'imu_latency': 'unavailable: trajectory text has no bag record '
                           'timestamp',
            'glim_lidar_stamp_semantics': 'scan start',
            'super_lio_lidar_stamp_semantics': 'scan end',
        },
        'data': {
            'matched_pose_count': int(len(matched)),
            'duration_sec': float(times[-1]),
            'filtered_replayed_lio_count': int(filtered_count),
            'localization_only_lio_count': int(len(estimates)),
            'imu_trajectory_pose_count': int(len(imu_trajectory)),
            'imu_trajectory_path': str(Path(args.imu_trajectory).resolve()),
        },
        'position_error_m': position_summary,
        'orientation_error_deg': orientation_summary,
        'position_jump_m_per_frame': jump_summary,
        'velocity_residual_mps': velocity_residual_summary,
        'comparisons': {
            ODOM_GT_COMPARISON: {
                'position_error_m': position_summary,
                'orientation_error_deg': orientation_summary,
                'position_jump_m_per_frame': jump_summary,
                'velocity_residual_mps': velocity_residual_summary,
                'event_counts': {
                    'position_jump': int(np.sum(jump_mask)),
                    'orientation_jump': int(np.sum(orientation_jump_mask)),
                    'position_error_episodes': int(len(contiguous_episodes(
                        error_mask, times, position_errors))),
                },
                'motion_dependency': {
                    'speed_vs_error_pearson': speed_pearson,
                    'speed_vs_error_spearman': speed_spearman,
                    'acceleration_vs_error_pearson': acceleration_pearson,
                    'acceleration_vs_error_spearman': acceleration_spearman,
                },
            },
            IMU_GT_COMPARISON: {
                'position_error_m': imu_position_summary,
                'orientation_error_deg': imu_orientation_summary,
                'position_jump_m_per_frame': imu_jump_summary,
                'velocity_residual_mps': imu_velocity_residual_summary,
                'event_counts': {
                    'position_jump': int(np.sum(imu_jump_mask)),
                    'orientation_jump': int(np.sum(
                        imu_orientation_jump_mask)),
                    'position_error_episodes': int(len(contiguous_episodes(
                        imu_error_mask, times, imu_position_errors))),
                },
                'motion_dependency': {
                    'speed_vs_error_pearson': imu_speed_pearson,
                    'speed_vs_error_spearman': imu_speed_spearman,
                    'acceleration_vs_error_pearson': (
                        imu_acceleration_pearson),
                    'acceleration_vs_error_spearman': (
                        imu_acceleration_spearman),
                },
            },
        },
        'latency_ms': {
            'GLIM': glim_latency_summary,
            'Super-LIO': super_latency_summary,
            'Super-LIO_steady_state': {
                **super_steady_latency_summary,
                'start_sec': args.latency_steady_state_start_sec,
            },
        },
        'thresholds': {
            'position_error_m': args.position_error_threshold_m,
            'position_jump_m': args.jump_threshold_m,
            'orientation_jump_deg': args.orientation_jump_threshold_deg,
            'latency_spike_ms': args.latency_spike_threshold_ms,
        },
        'event_counts': {
            'position_jump': int(np.sum(jump_mask)),
            'orientation_jump': int(np.sum(orientation_jump_mask)),
            'position_error_episodes': int(len(contiguous_episodes(
                error_mask, times, position_errors))),
            'latency_spike': int(np.sum(latency_spike_mask)),
            'latency_spike_startup': int(np.sum(
                latency_spike_mask & ~steady_latency_mask)),
            'latency_spike_steady_state': int(np.sum(
                steady_latency_spike_mask)),
            'imu_position_jump': int(np.sum(imu_jump_mask)),
            'imu_orientation_jump': int(np.sum(
                imu_orientation_jump_mask)),
            'imu_position_error_episodes': int(len(contiguous_episodes(
                imu_error_mask, times, imu_position_errors))),
        },
        'motion_dependency': {
            'speed_vs_error_pearson': speed_pearson,
            'speed_vs_error_spearman': speed_spearman,
            'acceleration_vs_error_pearson': acceleration_pearson,
            'acceleration_vs_error_spearman': acceleration_spearman,
            'speed_mps': percentile_summary(speeds),
            'acceleration_mps2': percentile_summary(accelerations),
        },
        'fixed_transform_map_from_glim': map_from_glim.tolist(),
        'artifacts': {
            'timeseries_csv': 'timeseries.csv',
            'latency_csv': 'latency.csv',
            'events_csv': 'events.csv',
            'motion_bins_csv': 'motion_bins.csv',
        },
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')

    markdown = f'''# Super-LIO against two GLIM ground truths

## 핵심 결과

- 비교 구간: **{times[-1]:.2f} s**, 동기 pose **{len(matched)}개**
- Super-LIO vs GLIM `odom_corrected` GT 위치오차 RMSE/p95/max: **{position_summary['rmse']*100:.2f}/{position_summary['p95']*100:.2f}/{position_summary['max']*100:.2f} cm**
- Super-LIO vs GLIM `traj_imu` GT 위치오차 RMSE/p95/max: **{imu_position_summary['rmse']*100:.2f}/{imu_position_summary['p95']*100:.2f}/{imu_position_summary['max']*100:.2f} cm**
- Super-LIO vs GLIM `odom_corrected` GT 자세오차 mean/p95/max: **{orientation_summary['mean']:.2f}/{orientation_summary['p95']:.2f}/{orientation_summary['max']:.2f}°**
- Super-LIO vs GLIM `traj_imu` GT 자세오차 mean/p95/max: **{imu_orientation_summary['mean']:.2f}/{imu_orientation_summary['p95']:.2f}/{imu_orientation_summary['max']:.2f}°**
- 관측 processing latency: GLIM median/p95/max **{glim_latency_summary['median']:.2f}/{glim_latency_summary['p95']:.2f}/{glim_latency_summary['max']:.2f} ms**, Super-LIO **{super_latency_summary['median']:.2f}/{super_latency_summary['p95']:.2f}/{super_latency_summary['max']:.2f} ms**
- Super-LIO 정상 주행({args.latency_steady_state_start_sec:.1f} s 이후) latency median/p95/max **{super_steady_latency_summary['median']:.2f}/{super_steady_latency_summary['p95']:.2f}/{super_steady_latency_summary['max']:.2f} ms**
- 위치 jump(>{args.jump_threshold_m*100:.1f} cm/frame): `odom_corrected` GT 기준 **{int(np.sum(jump_mask))}회**, `traj_imu` GT 기준 **{int(np.sum(imu_jump_mask))}회**
- 자세 jump(>{args.orientation_jump_threshold_deg:.1f}°/frame): `odom_corrected` GT 기준 **{int(np.sum(orientation_jump_mask))}회**, `traj_imu` GT 기준 **{int(np.sum(imu_orientation_jump_mask))}회**
- 위치오차 {args.position_error_threshold_m*100:.1f} cm 초과 episode: `odom_corrected` GT 기준 **{len(contiguous_episodes(error_mask, times, position_errors))}회**, `traj_imu` GT 기준 **{len(contiguous_episodes(imu_error_mask, times, imu_position_errors))}회**
- Super-LIO latency {args.latency_spike_threshold_ms:.0f} ms 초과: 초기화 **{int(np.sum(latency_spike_mask & ~steady_latency_mask))}회**, 정상 주행 **{int(np.sum(steady_latency_spike_mask))}회**
- 속도-오차 Pearson/Spearman: `odom_corrected` GT 기준 **{format_correlation(speed_pearson)} / {format_correlation(speed_spearman)}**, `traj_imu` GT 기준 **{format_correlation(imu_speed_pearson)} / {format_correlation(imu_speed_spearman)}**
- 가속도-오차 Pearson/Spearman: `odom_corrected` GT 기준 **{format_correlation(acceleration_pearson)} / {format_correlation(acceleration_spearman)}**, `traj_imu` GT 기준 **{format_correlation(imu_acceleration_pearson)} / {format_correlation(imu_acceleration_spearman)}**

## 방법

- GLIM `odom_corrected`와 GLIM `traj_imu`를 각각 ground truth로 가정해 Super-LIO를 별도로 평가했습니다.
- `odom_corrected`는 TXT와 동일 pose가 들어 있는 reference bag의 `/glim_ros/odom_corrected`를 사용해 sensor `header.stamp`로 동기화했습니다.
- 전체 궤적 최적 정렬 없이, prebuilt-map bag과 GLIM bag의 동일 센서 시각 pose 한 쌍으로 구한 고정 `T_map_glim`을 GLIM GT와 `traj_imu`에 똑같이 적용했습니다.
- 세 trajectory 모두 원래 sensor timestamp로 동기화했으며 fitted time offset은 사용하지 않았습니다.
- latency는 같은 bag 안에서 LiDAR scan 기록 시각부터 odometry 출력 기록 시각까지의 관측값입니다. ROS executor/recorder scheduling이 일부 포함됩니다.
- latency는 GT 선택과 무관한 출력 지연입니다. `traj_imu.txt` 자체의 processing latency는 bag record timestamp가 없어 계산할 수 없습니다.
- Super-LIO bag에 섞인 재생 원본 `/lio/odom` {filtered_count}개는 제외했습니다.

## 그래프

![Error time series](01_error_timeseries.png)

![Latency](02_latency.png)

![Motion dependency](03_motion_dependency.png)

![Jump events](04_jump_events.png)

![Motion bins](05_motion_bins.png)
'''
    (output_dir / 'summary.md').write_text(markdown)
    print(json.dumps({
        'output_dir': str(output_dir),
        'super_vs_odom_gt_position_rmse_cm': (
            position_summary['rmse'] * 100.0),
        'super_vs_odom_gt_position_p95_cm': (
            position_summary['p95'] * 100.0),
        'super_vs_odom_gt_position_max_cm': (
            position_summary['max'] * 100.0),
        'super_latency_median_ms': super_latency_summary['median'],
        'super_latency_p95_ms': super_latency_summary['p95'],
        'super_vs_odom_gt_position_jump_count': int(np.sum(jump_mask)),
        'super_vs_imu_gt_position_rmse_cm': (
            imu_position_summary['rmse'] * 100.0),
        'super_vs_imu_gt_position_p95_cm': (
            imu_position_summary['p95'] * 100.0),
        'super_vs_imu_gt_position_max_cm': (
            imu_position_summary['max'] * 100.0),
        'super_vs_imu_gt_position_jump_count': int(np.sum(imu_jump_mask)),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-frame-bag', required=True)
    parser.add_argument('--reference-bag', required=True)
    parser.add_argument('--estimated-bag', required=True)
    parser.add_argument('--imu-trajectory', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--map-frame-topic', default='/odometry/odom')
    parser.add_argument(
        '--calibration-reference-topic', default='/odometry/odom')
    parser.add_argument(
        '--reference-topic', default='/glim_ros/odom_corrected')
    parser.add_argument('--estimated-topic', default='/lio/odom')
    parser.add_argument('--replayed-input-topic', default='/lio/odom')
    parser.add_argument('--position-error-threshold-m', type=float, default=0.05)
    parser.add_argument('--jump-threshold-m', type=float, default=0.05)
    parser.add_argument(
        '--orientation-jump-threshold-deg', type=float, default=5.0)
    parser.add_argument('--latency-spike-threshold-ms', type=float, default=50.0)
    parser.add_argument(
        '--latency-steady-state-start-sec', type=float, default=1.0)
    parser.add_argument('--motion-smoothing-samples', type=int, default=9)
    parser.add_argument('--rolling-window-sec', type=float, default=1.0)
    analyze(parser.parse_args())


if __name__ == '__main__':
    main()

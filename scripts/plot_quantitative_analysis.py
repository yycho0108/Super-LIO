#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    'glim': '#ff23ff',
    'super': '#19ff3c',
    'imu': '#00a6ff',
    'error': '#ff5a5f',
    'blue': '#38bdf8',
    'orange': '#f59e0b',
    'gray': '#94a3b8',
}
ODOM_GT_COMPARISON = 'Super-LIO vs GLIM odom_corrected'
IMU_GT_COMPARISON = 'Super-LIO vs GLIM traj_imu'


def style_axes(axes):
    for axis in np.atleast_1d(axes).ravel():
        axis.grid(True, alpha=0.25)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def error_timeseries(data, output):
    time = data['time_sec']
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for prefix, label, color in (
            ('', 'GT: odom_corrected', COLORS['super']),
            ('imu_', 'GT: traj_imu', COLORS['imu'])):
        axes[0].plot(time, data[f'{prefix}position_error_m'] * 100,
                     color=color, linewidth=0.9, alpha=0.35,
                     label=f'{label} error')
        axes[0].plot(time, data[f'{prefix}rolling_rmse_m'] * 100,
                     color=color, linewidth=2.0,
                     label=f'{label} 1 s RMSE')
        axes[0].plot(time, data[f'{prefix}cumulative_rmse_m'] * 100,
                     color=color, linewidth=1.2, linestyle=':',
                     label=f'{label} cumulative')
    axes[0].set_ylabel('Position error [cm]')
    axes[0].legend(ncol=3, loc='upper right')
    for prefix, algorithm, color in (
            ('', 'odom GT', COLORS['super']),
            ('imu_', 'imu GT', COLORS['imu'])):
        for axis_name, style in (('x', '-'), ('y', '--'), ('z', ':')):
            axes[1].plot(
                time, data[f'{prefix}error_{axis_name}_m'] * 100,
                label=f'{algorithm} {axis_name}', color=color,
                linestyle=style, linewidth=1.0)
    axes[1].axhline(0, color='black', linewidth=0.7)
    axes[1].set_ylabel('Signed error [cm]')
    axes[1].legend(ncol=3, loc='upper right')
    for prefix, label, color in (
            ('', 'GT: odom_corrected', COLORS['super']),
            ('imu_', 'GT: traj_imu', COLORS['imu'])):
        axes[2].plot(time, data[f'{prefix}orientation_error_deg'],
                     color=color, label=f'{label} 3D')
        axes[2].plot(time, data[f'{prefix}yaw_error_deg'],
                     color=color, linestyle=':', label=f'{label} yaw')
    axes[2].set_ylabel('Orientation error [deg]')
    axes[2].set_xlabel('Localization elapsed time [s]')
    axes[2].legend(ncol=2, loc='upper right')
    axes[0].set_title(
        'Super-LIO error against two GLIM ground-truth candidates')
    style_axes(axes)
    save(fig, output / '01_error_timeseries.png')


def latency_plot(latency, summary, output):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    for algorithm, color in (
            ('GLIM', COLORS['glim']), ('Super-LIO', COLORS['super'])):
        current = latency[latency['algorithm'] == algorithm]
        axes[0].plot(current['time_sec'], current['latency_ms'],
                     '.', markersize=2.5, alpha=0.8,
                     color=color, label=algorithm)
        axes[1].hist(current['latency_ms'], bins=45, alpha=0.55,
                     color=color, label=algorithm)
        median = summary['latency_ms'][algorithm]['median']
        axes[1].axvline(median, color=color, linestyle='--', linewidth=1.4)
    threshold = summary['thresholds']['latency_spike_ms']
    steady_start = summary['latency_ms']['Super-LIO_steady_state']['start_sec']
    axes[0].axhline(threshold, color='#ef4444', linestyle='--',
                    linewidth=1.2, label=f'spike threshold {threshold:g} ms')
    axes[0].axvline(steady_start, color='#64748b', linestyle=':',
                    linewidth=1.2, label=f'steady state from {steady_start:g} s')
    axes[0].set_ylabel('Observed processing latency [ms]')
    axes[0].set_xlabel('Algorithm-local elapsed time [s]')
    axes[0].set_title('LiDAR record → odometry record latency')
    axes[0].text(
        0.01, 0.90,
        'GLIM traj_imu: unavailable (TXT has no bag record timestamp)',
        transform=axes[0].transAxes, color=COLORS['imu'], fontsize=9)
    axes[0].legend(ncol=3, loc='upper right')
    axes[1].set_xlabel('Observed processing latency [ms]')
    axes[1].set_ylabel('Samples')
    axes[1].legend()
    style_axes(axes)
    save(fig, output / '02_latency.png')


def motion_dependency(data, output):
    time = data['time_sec']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for prefix, label, color in (
            ('', 'odom GT', COLORS['super']),
            ('imu_', 'traj_imu GT', COLORS['imu'])):
        axes[0, 0].plot(time, data[f'{prefix}speed_mps'], color=color,
                        label=f'{label} speed')
        axes[0, 0].plot(
            time, data[f'{prefix}acceleration_mps2'], color=color,
            linestyle=':', alpha=0.8, label=f'{label} acceleration')
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Speed [m/s], acceleration [m/s²]')
    axes[0, 0].legend()
    axes[0, 0].set_title('Robot motion from smoothed GLIM trajectory')

    for prefix, label, color in (
            ('', 'GT: odom_corrected', COLORS['super']),
            ('imu_', 'GT: traj_imu', COLORS['imu'])):
        axes[0, 1].plot(time, data[f'{prefix}position_error_m'] * 100,
                        color=color, label=f'{label} position error')
        axes[0, 1].plot(
            time, data[f'{prefix}velocity_residual_mps'] * 100,
            color=color, linestyle=':', alpha=0.8,
            label=f'{label} velocity residual')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Error [cm], velocity residual [cm/s]')
    axes[0, 1].legend()
    axes[0, 1].set_title('Localization stability over motion')

    for prefix, speed_column, label, color in (
            ('', 'speed_mps', 'GT: odom_corrected', COLORS['super']),
            ('imu_', 'imu_speed_mps', 'GT: traj_imu', COLORS['imu'])):
        axes[1, 0].scatter(
            data[speed_column], data[f'{prefix}position_error_m'] * 100,
            color=color, s=14, alpha=0.45, label=label)
    axes[1, 0].set_xlabel('GT speed [m/s]')
    axes[1, 0].set_ylabel('Position error [cm]')
    axes[1, 0].set_title('Error vs speed')
    axes[1, 0].legend()

    for prefix, acceleration_column, label, color in (
            ('', 'acceleration_mps2', 'GT: odom_corrected',
             COLORS['super']),
            ('imu_', 'imu_acceleration_mps2', 'GT: traj_imu',
             COLORS['imu'])):
        axes[1, 1].scatter(
            data[acceleration_column],
            data[f'{prefix}position_error_m'] * 100,
            color=color, s=14, alpha=0.45, label=label)
    axes[1, 1].set_xlabel('GT acceleration [m/s²]')
    axes[1, 1].set_ylabel('Position error [cm]')
    axes[1, 1].set_title('Error vs acceleration')
    axes[1, 1].legend()
    style_axes(axes)
    save(fig, output / '03_motion_dependency.png')


def jump_plot(data, summary, output):
    time = data['time_sec']
    jump_threshold = summary['thresholds']['position_jump_m'] * 100
    orientation_threshold = summary['thresholds']['orientation_jump_deg']
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    masks = {}
    for prefix, label, color in (
            ('', 'GT: odom_corrected', COLORS['super']),
            ('imu_', 'GT: traj_imu', COLORS['imu'])):
        jump_mask = (
            data[f'{prefix}position_jump_m'].to_numpy() * 100 >
            jump_threshold)
        orientation_mask = (
            data[f'{prefix}orientation_jump_deg'].to_numpy() >
            orientation_threshold)
        masks[prefix] = jump_mask
        axes[0].plot(time, data[f'{prefix}position_jump_m'] * 100,
                     color=color, linewidth=1.0, label=label)
        axes[0].scatter(
            time[jump_mask],
            data.loc[jump_mask, f'{prefix}position_jump_m'] * 100,
            color=color, edgecolor='black', s=35, zorder=3)
        axes[1].plot(time, data[f'{prefix}orientation_jump_deg'],
                     color=color, linewidth=1.0, label=label)
        axes[1].scatter(
            time[orientation_mask],
            data.loc[orientation_mask, f'{prefix}orientation_jump_deg'],
            color=color, edgecolor='black', s=35, zorder=3)
    axes[0].axhline(jump_threshold, color='#991b1b', linestyle='--',
                    label=f'threshold {jump_threshold:.1f} cm/frame')
    axes[0].set_ylabel('Error-vector step [cm/frame]')
    axes[0].legend()
    axes[0].set_title('Localization jump diagnostics')

    axes[1].axhline(orientation_threshold, color='#991b1b', linestyle='--',
                    label=f'threshold {orientation_threshold:.1f}°/frame')
    axes[1].set_ylabel('Orientation-error step [deg/frame]')
    axes[1].legend()

    axes[2].plot(data['gt_x_m'], data['gt_y_m'],
                 color=COLORS['glim'], linewidth=2.5,
                 label='GLIM odom_corrected GT')
    axes[2].plot(data['super_x_m'], data['super_y_m'],
                 color=COLORS['super'], linewidth=1.5, label='Super-LIO')
    axes[2].plot(data['imu_x_m'], data['imu_y_m'],
                 color=COLORS['imu'], linewidth=1.5,
                 label='GLIM traj_imu GT')
    for prefix, x_column, y_column, color in (
            ('', 'super_x_m', 'super_y_m', COLORS['super']),
            ('imu_', 'super_x_m', 'super_y_m', COLORS['imu'])):
        axes[2].scatter(
            data.loc[masks[prefix], x_column],
            data.loc[masks[prefix], y_column], color=color,
            edgecolor='black', s=45, zorder=4)
    axes[2].set_xlabel('x [m]')
    axes[2].set_ylabel('y [m]')
    axes[2].axis('equal')
    axes[2].legend()
    style_axes(axes)
    save(fig, output / '04_jump_events.png')


def motion_bins_plot(bins, output):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for axis, metric, title, unit in (
            (axes[0], 'speed_mps', 'Error by speed quartile', 'm/s'),
            (axes[1], 'acceleration_mps2',
             'Error by acceleration quartile', 'm/s²')):
        current = bins[bins['metric'] == metric]
        template = current[
            current['algorithm'] == ODOM_GT_COMPARISON].reset_index(drop=True)
        positions = np.arange(len(template))
        labels = ['Q1\n(low)', 'Q2', 'Q3', 'Q4\n(high)'][:len(template)]
        for offset, algorithm, label, color in (
                (-0.19, ODOM_GT_COMPARISON, 'GT: odom_corrected',
                 COLORS['super']),
                (0.19, IMU_GT_COMPARISON, 'GT: traj_imu', COLORS['imu'])):
            values = current[current['algorithm'] == algorithm].sort_values(
                'bin_index')
            axis.bar(positions + offset, values['rmse_m'] * 100, width=0.34,
                     color=color, alpha=0.75, label=f'{label} RMSE')
            axis.plot(positions + offset, values['p95_m'] * 100, 'o--',
                      color=color, linewidth=1.4,
                      label=f'{label} p95')
        axis.set_xticks(positions, labels)
        axis.set_ylabel('Position error [cm]')
        axis.set_title(title)
        axis.legend()
    style_axes(axes)
    save(fig, output / '05_motion_bins.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    args = parser.parse_args()
    output = Path(args.input_dir)
    data = pd.read_csv(output / 'timeseries.csv')
    latency = pd.read_csv(output / 'latency.csv')
    bins = pd.read_csv(output / 'motion_bins.csv')
    summary = json.loads((output / 'summary.json').read_text())
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 13,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
    })
    error_timeseries(data, output)
    latency_plot(latency, summary, output)
    motion_dependency(data, output)
    jump_plot(data, summary, output)
    motion_bins_plot(bins, output)
    print(f'Wrote quantitative plots to {output}')


if __name__ == '__main__':
    main()

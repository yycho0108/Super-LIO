# Super-LIO relocation parameter optimization

This pipeline keeps `relocation_real_20260802.yaml` unchanged. Every trial gets
its own generated YAML, clean `/lio/odom` bag, logs, and JSON metrics. Trial bags
are always written below `/workspace/rosbag`; configs and metrics stay in the
repository-side `localization_replay/optimization_runs` directory.

## Fair comparison definition

- Time: `/lio/odom.header.stamp` and the first column of GLIM `traj_imu.txt`.
  There is no fitted time offset.
- Pose type: Livox IMU pose against GLIM IMU pose.
- Space: one `T_map_glim` is computed from the original source bag's
  `/lio/odom` and GLIM at their first common timestamp. That same transform is
  applied to every candidate. It is never re-fit per frame or per candidate.
- The score reports position mean/RMSE/p95/max and signed/absolute XYZ errors.
  Absolute Z max has the largest individual weight. Coverage below 90% receives
  a large penalty, so a truncated or failed run cannot win.

GLIM is a comparison reference, not surveyed ground truth. A winner should be
cross-validated on multiple bags before being accepted for the robot.

## Commands

First inspect a small plan (no ROS processes are started):

```bash
cd /home/dongjae/ws/src/Super-LIO
localization_replay/run_relocation_optimization.sh \
  --bag /workspace/rosbag/rosbag2_2026_08_06-07_32_44 \
  --study aug06_073244_smoke --profile imu --max-trials 3
```

Run the IMU-noise coordinate search at the same 1.0 playback rate used for the
manual experiment:

```bash
cd /home/dongjae/ws/src/Super-LIO
localization_replay/run_relocation_optimization.sh \
  --bag /workspace/rosbag/rosbag2_2026_08_06-07_32_44 \
  --study aug06_073244_imu --profile imu --execute
```

After screening IMU parameters, use `--profile core` with a new study name.
For stronger generalization, repeat `--bag` for two or more source bags; every
parameter set is then replayed on every listed bag and ranked by mean/worst-case
score.

The winner is written to both:

- `localization_replay/optimization_runs/STUDY/relocation_optimized.yaml`
- `src/super_lio/config/relocation_optimized.yaml`

If `--max-trials` stops a study early, only the study-local winner is written;
the shared optimized config is intentionally left untouched. Add
`--install-partial-best` only when that partial winner is deliberately wanted.

The original config is never overwritten. View any compact trial in Viser with
the existing command; robot joint state, TF, augmented map, and static map come
from the matching original bag while the trajectory comes from the clean trial:

```bash
localization_replay/run_localization_replay.sh \
  /workspace/rosbag/superlio_2026_08_06-07_32_44_opt_STUDY_t000
```

The sweep records only `/lio/odom`. After choosing a winner, rerun that YAML
once with the normal full recording command when a candidate-specific
`/lio/cloud_world` bag is needed.

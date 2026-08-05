# 2026-08-04 GLIM vs Super LIO replay

네 개 실험에 같은 frame/URDF/error/Viser 로직을 사용하는 실행기입니다.

## 새 컴퓨터에서 준비

ROS 2 Humble과 bag의 `unitree_hg/msg/LowState`를 읽을 수 있는 Unitree ROS 2
overlay가 설치되어 있어야 합니다. Unitree overlay가 현재 셸에 source되어 있거나
기본 위치인 `~/unitree_ros2/cyclonedds_ws/install/setup.bash`에 있으면 됩니다.
다른 위치라면 다음처럼 지정합니다.

```bash
export UNITREE_ROS_SETUP=/path/to/unitree_ros2/install/setup.bash
./setup_env.sh
```

`setup_env.sh`는 프로젝트 안에 `.venv`를 새로 만들고 Viser 등 Python 의존성을
설치합니다. `.venv`는 컴퓨터와 절대경로에 종속되므로 Git에 포함하지 않습니다.
실행 스크립트도 `.venv`가 없으면 자동으로 생성합니다.

PCD 지도, G1 URDF와 실제 사용되는 STL mesh는 `assets/`에 포함되어 있으므로
별도의 `~/ws`나 MuJoCo asset 디렉터리는 필요하지 않습니다.

```bash
cd /path/to/localization_replay
./run_superlio_062309.sh
./run_superlio_063004.sh
./run_superlio_064152.sh
./run_superlio_064922.sh
```

각 실행기는 서로 다른 기본 포트를 사용하므로 동시에 실행할 수도 있습니다.

기본적으로 `$HOME/rosbag` 아래에서 원본 bag과 GLIM dump를 찾습니다. 다른 상위
디렉터리에 있다면 다음처럼 지정합니다.

```bash
./run_superlio_062309.sh --data-root /path/to/rosbag-root
```

각 입력을 따로 지정할 수도 있습니다.

```bash
./run_superlio_062309.sh \
  --bag /path/to/evaluation-bag \
  --source-bag /path/to/original-bag \
  --glim-trajectory /path/to/traj_imu.txt
```

`traj_lidar.txt`는 `traj_imu.txt`와 같은 디렉터리에 있어야 합니다.

| run | 원본 실험 bag | GLIM `traj_imu.txt` | port | 공통 sensor 구간 |
|---|---|---|---:|---:|
| 06:23:09 | `rosbag2_2026_08_04-06_23_09` | `glim_rosbag2_2026_08_04-06_23_09` | 8082 | 51.834 s |
| 06:30:04 | `rosbag2_2026_08_04-06_30_04` | `glim_rosbag2_2026_08_04-06_30_04` | 8081 | 154.167 s |
| 06:41:52 | `rosbag2_2026_08_04-06_41_52` | `glim_rosbag2_2026_08_04-06_41_52` | 8083 | 133.533 s |
| 06:49:22 | `rosbag2_2026_08_04-06_49_22` | `glim_rosbag2_2026_08_04-06_49_22` | 8084 | 52.967 s |

## 실제 사용 데이터

네 원본 실험 bag에 실제 주행 당시 Super-LIO `/lio/odom`이 이미 기록되어 있습니다.
따라서 기본 비교는 다음 IMU pose 두 개입니다.

- Super LIO: 원본 실험 bag의 recorded live `/lio/odom` `header.stamp`
- GLIM: 같은 run의 GLIM dump `traj_imu.txt`
- GLIM→map frame anchor: 원본 bag의 `/odometry/odom`과 같은 dump의
  `traj_lidar.txt`

`superlio_062309/063004/064152/064922`는 나중 offline replay 때 만든 재기록
bag입니다. 여기에는 원본 `/lio/odom`과 새 재실행 출력이 같은 topic으로 섞여
있으므로 기본 평가에서 완전히 제외합니다. 특히 `superlio_064152`의 재실행 출력만
약 89.4초부터 발산했고, 실제 실험 당시 원본 `/lio/odom`은 정상입니다.

경로를 바꾸려면 명시적으로 지정할 수 있습니다.

```bash
./run_superlio_063004.sh \
  --bag /path/to/original_rosbag \
  --glim-trajectory /path/to/traj_imu.txt \
  --source-bag /path/to/original_rosbag
```

추후 선택 bag에 `/glim_ros/odom_corrected`가 들어 있으면 그 topic을 우선 사용하고
txt fallback은 사용하지 않습니다. LIO는 `/lio/imu/odom`이 있으면 우선하고, 없으면
현재처럼 `/lio/odom`을 사용합니다.

## 시간 및 공간 비교

bag 경과시간이 아니라 두 결과의 절대 sensor timestamp를 사용합니다. GLIM의 실제
pose timestamp 각각에서 Super LIO를 보간해 position 및 SO(3) orientation error를
계산합니다.

공간 정렬은 각 run의 원본 bag과 GLIM dump가 처음 함께 존재하는 sensor timestamp
하나에서 아래 고정 SE(3)를 계산합니다.

```text
T_MAP_GLIM = T_MAP_LIDAR(/odometry/odom, t_cal)
             · inverse(T_GLIM_LIDAR(traj_lidar.txt, t_cal))
T_MAP_IMU(t) = T_MAP_GLIM · T_GLIM_IMU(traj_imu.txt, t)
```

원본 `/odometry/odom`은 timestamp가 같은 `/lio/odom`에 고정
`T_IMU_LIDAR=(-0.011,-0.02329,+0.04412)`를 곱한 stream입니다. 따라서 이 변환은
독립 측량 calibration이 아니라 첫 pose를 같게 두는 **고정 start-frame anchor**입니다.
즉 초기 절대 localization bias는 평가하지 않고, 이후 두 궤적의 상대적인 불일치를
평가합니다. trajectory 전체 fitting이나 시각별 정합은 하지 않으며 모든 pose에 같은
행렬 하나만 적용합니다.

| run | translation xyz [m] | RPY [deg] |
|---|---|---|
| 06:23:09 | (-4.0864, -1.5918, -0.0549) | (-0.774, -0.112, -160.843) |
| 06:30:04 | (-3.9982, -1.9055, -0.0329) | (-0.908, -1.045, 146.518) |
| 06:41:52 | (-3.6020, -1.8734, -0.0270) | (0.331, -1.333, 107.966) |
| 06:49:22 | (-3.1859, -2.2146, -0.7111) | (-14.716, 5.927, 44.999) |

정확한 4×4 행렬은 dry-run 출력과 Viser의 `Fixed start-frame anchor`
폴더에서 확인할 수 있습니다. 원본 GLIM 좌표를 보려면 `--no-spatial-align`을
사용합니다.

## UI 기능

- static `dump7_xyzi_clean.pcd` map과 bag `/augmented_map`
- GLIM/Super LIO 전체 path, 현재 IMU frame 및 이동 marker
- 같은 `/lowstate` joint를 가진 두 개의 독립적인 반투명 G1 robot
- robot별 visibility checkbox
- position/orientation error graph의 이동 점과 노란 cursor
- error graph 클릭 및 좌우 drag로 slider, robot, pose, joint, bag map 동기 이동
- trajectory XYZ/RPY plot과 선택 joint q/dq/torque plot
- live ROS `/augmented_map` overlay
- 실제 29-DoF G1 URDF와 `/lowstate`를 사용한 robot FK
- 모든 point cloud 기본 크기 `0.01`; UI slider 변경값은 timestamp 이동 후에도 유지
- grid/scene origin은 초기 GLIM robot 발밑, 초기 camera target은 그 위 robot 중심
- `Space`: play/pause toggle
- `←` / `→`: 재생을 멈추고 정확한 이전/다음 `/lio/odom.header.stamp` frame으로 이동
- Playback 폴더의 `Previous frame` / `Next frame` 버튼으로도 동일한 frame stepping

키보드 단축키는 3D 화면이나 빈 영역에 focus가 있을 때 동작합니다. 숫자 입력창을
편집 중일 때는 입력을 방해하지 않도록 단축키를 무시합니다. `current /lio/odom
frame`과 `frame stepping` 항목에서 현재 frame 번호와 exact timestamp 여부를 확인할
수 있습니다.

## Robot pose의 frame 체인

ROS `TransformStamped(parent, child)`의 값은 child 좌표를 parent 좌표로 옮기는
`T_parent_child`입니다. 반면
`yourdfpy.get_transform(frame_to, frame_from)`은 `T_frame_from_frame_to`를 반환합니다.
실제 URDF의 `pelvis→imu_in_pelvis=(0.04525, 0, -0.08339)`로 확인하면 다음처럼
서로 inverse입니다.

```text
get_transform("imu_in_pelvis", "pelvis") = T_pelvis_imu
get_transform("pelvis", "imu_in_pelvis") = T_imu_pelvis
```

Super-LIO source에서 `/lio/odom`은 lidar pose가 아니라 lidar 주기로 출력되는
`T_world_imu`입니다. 원본 실험 bag의 named physical TF와
`T_imu_lidar=(-0.011,-0.02329,+0.04412)`는 일치합니다.

대신 bag/robot_description의 이름이 명확한 물리 프레임을 사용합니다.

```text
T_map_pelvis
 = T_map_world
 · T_world_mid360_link_IMU          (/lio/odom)
 · inverse(T_mid360_link_frame_mid360_link_IMU)
 · inverse(T_mid360_link_mid360_link_frame)
 · T_mid360_link_pelvis             (URDF FK + /lowstate)
```

`mid360_link→mid360_link_frame`를 inverse하는 것이 수학적으로 맞습니다. 회전이 거의
180도라 원행렬과 inverse가 수치상 비슷해 보일 뿐, 필요한 방향은
`T_mid360_link_frame_mid360_link`입니다.

기본 URDF는 실제 로봇 description인
`g1_29dof_rev_1_0.urdf`입니다. `/lowstate.motor_state[0:29]` 순서는 Unitree의
공식 G1 joint index와 동일하며, 같은 시각 bag `/tf`의 29개 joint와 비교했을 때
최대 translation 차이 0 mm, rotation 차이 약 0.0014도였습니다.

## 높이 검증

어떤 map/trajectory/robot z-offset도 적용하지 않습니다. `dump7_xyzi_clean.pcd`의
dominant floor는 `z=-1.325 m`이고 foot mesh의 최저점은 ankle-roll origin에서
`-0.035409 m`입니다.

Viser의 grid/origin은 좌표 데이터를 이동시키지 않고, 첫 공통 시각의 GLIM robot
양쪽 ankle과 sole mesh offset으로 계산한 시각화 기준점입니다.

| run | 초기 grid/origin z | dominant floor 대비 |
|---|---:|---:|
| 06:23:09 | -1.3133 m | +1.17 cm |
| 06:30:04 | -1.3129 m | +1.21 cm |
| 06:41:52 | -1.3237 m | +0.13 cm |
| 06:49:22 | -1.3258 m | -0.08 cm |

어떤 run에도 floor fitting이나 z-offset은 적용하지 않았습니다. grid와 camera만 이
visual-only origin을 사용하고 map, trajectory, robot pose 값은 그대로 유지합니다.

dry-run의 원본 live Super-LIO 비교 결과는 다음과 같습니다.

| run | position mean/RMSE/max [m] | orientation mean/RMSE/max [deg] |
|---|---|---|
| 06:23:09 | 0.0155 / 0.0174 / 0.0569 | 0.347 / 0.392 / 1.599 |
| 06:30:04 | 0.0216 / 0.0244 / 0.1749 | 0.379 / 0.430 / 2.275 |
| 06:41:52 | 0.0169 / 0.0205 / 0.0993 | 0.380 / 0.418 / 1.679 |
| 06:49:22 | 0.0258 / 0.0267 / 0.0626 | 0.393 / 0.431 / 1.298 |

네 실제 실험 stream 모두 발산하지 않습니다. 위 수치는 첫 pose frame anchor 이후의
상대 궤적 차이이며, 독립 측량 기반 absolute localization error는 아닙니다.

## 점검 및 옵션

서버를 띄우지 않고 데이터/정렬/error 통계만 검사할 수 있습니다.

```bash
./run_superlio_062309.sh --dry-run
./run_superlio_063004.sh --dry-run
./run_superlio_064152.sh --dry-run
./run_superlio_064922.sh --dry-run
./run_superlio_063004.sh --port 8091
./run_superlio_063004.sh --no-live-aug-map
./run_superlio_063004.sh --no-static-map
./run_superlio_063004.sh --no-robot
```

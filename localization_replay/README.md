# G1 localization replay

새 2026-08-04 06:30:04 bag의 GLIM/Super-LIO 두 경로 전용 실행기는
[`README_superlio_063004.md`](README_superlio_063004.md)를 참고합니다.

## 환경과 포함 asset

PCD 지도와 G1 URDF/STL mesh는 `assets/`에 포함되어 있습니다. Python 가상환경은
복사하거나 Git에 올리지 않고 각 컴퓨터에서 생성합니다.

```bash
./setup_env.sh
```

ROS 2 Humble과 Unitree ROS 2 메시지 패키지는 시스템에 설치되어 있어야 합니다.
자세한 데이터 경로와 환경변수는 `README_superlio_063004.md`를 참고합니다.

세 개의 localization trajectory와 두 개의 ROS 2 bag `/lowstate`·`/augmented_map`, live ROS `/augmented_map`, G1 URDF, PCD map을 하나의 Viser UI에서 동기 재생합니다.

## 실행

```bash
cd /path/to/localization_replay
./run.sh
```

터미널에 표시되는 `http://localhost:8080`을 브라우저에서 엽니다. `Exact same-timestamp error` 그래프를 클릭하거나 그래프 위에서 마우스를 좌우로 drag할 수 있고, `exact common timestamp offset [s]` 슬라이더도 같은 timeline을 제어합니다.

첫 실행에서는 두 replay bag의 약 72,000개 lowstate와 TF, Super-LIO evaluation bag의 odometry를 읽고 URDF mesh를 올리므로 서버 URL이 나오기까지 이 장비 기준 약 30초 걸립니다.

- 세 trajectory의 현재 IMU pose와 전체 path
- 각 trajectory pose에 놓인 G1 robot의 29개 low-state joint와 robot별 visibility checkbox
- 08-02/08-03 bag의 slider 동기 `/augmented_map` 및 live ROS `/augmented_map`
- trajectory별 XYZ/RPY plot의 이동 점과 노란색 time cursor
- 동일 sensor timestamp에서 계산한 세 pair의 position/orientation error plot
- 선택한 관절의 q, dq, estimated torque plot과 현재 값

## 공통 timestamp

bag 시작 후 경과시간을 억지로 맞추지 않습니다. 모든 pose를 원본 센서의 공통 `header.stamp` clock으로 변환합니다.

- GLIM IMU: txt timestamp가 원본 sensor/header time이므로 그대로 사용
- Super LIO: evaluation bag의 record time을 `/lio/odom` sensor header clock으로 변환하고, prebuilt-map bag에서 재생된 기존 pose 869개를 제외한 localization 출력 624개만 사용
- GLIM corrected: txt의 bag record time을 `/glim_ros/odom_corrected`의 대응 `header.stamp`로 변환
- 08-02/08-03 lowstate, TF, augmented map: 선택된 sensor timestamp를 각 bag record clock으로 역변환한 후 조회

세 결과가 모두 존재하는 정확한 공통 구간만 재생합니다.

```text
1785674499.225155115 .. 1785674519.929036379
duration: 20.703881 s
```

Super LIO의 현재 raw→sensor median offset은 약 `-88072.815482 s`입니다. 일반적으로 evaluation bag으로 직접 변환되므로 수동 설정은 필요하지 않습니다. 필요하면 직접 덮어쓸 수 있습니다.

```bash
./run.sh --super-time-offset -88072.815321
```

## 입력 포맷과 기본 경로

| source | 기본 경로 | quaternion 열 순서 |
|---|---|---|
| GLIM IMU | `<data-root>/Downloads/glim_dump8/traj_imu.txt` | `qx qy qz qw` |
| Super LIO | `<data-root>/super_lio_trajectory.txt` | `qw qx qy qz` |
| GLIM corrected | `<data-root>/glim_ros_odom_corrected_trajectory.txt` | `qw qx qy qz` |

사용자가 처음 적어준 두 파일명에는 각각 확장자 누락/철자 차이가 있어, 디스크에서 확인된 위 경로를 기본값으로 사용했습니다.

모든 경로는 CLI에서 변경할 수 있습니다.

```bash
./run.sh --help
./run.sh --data-root /path/to/data-root
./run.sh --port 8090 --max-map-points 500000
./run.sh --aug-map-point-size 0.02
./run.sh --no-robot
./run.sh --no-live-aug-map
./run.sh --live-aug-map-topic /my_augmented_map
./run.sh --no-glim-spatial-align
./run.sh --dry-run
```

## 고정 GLIM spatial alignment

평가 대상 trajectory를 정렬 기준으로 사용하지 않습니다. prebuilt-map bag과 GLIM bag에 공통으로 들어 있는 `/odometry/odom`의 동일 sensor timestamp pose 한 쌍에서 한 개의 고정 SE(3)를 계산합니다.

```text
t_cal = 1785674487.628974438
T_map_glim = T_map_odom_08-02(t_cal) · inverse(T_glim_odom_08-03(t_cal))
T_map_glimPose(t) = T_map_glim · T_glimPose(t), for every t
```

이 `T_align` 하나를 GLIM IMU와 corrected GLIM 모두에 그대로 적용합니다. corrected GLIM만 따로 시작점에 맞추지 않으므로 두 GLIM 출력 사이의 원래 위치/자세 차이는 보존됩니다. 현재 데이터에서 계산된 값은 다음과 같습니다.

```text
[[-0.981463 -0.191571  0.005608 -4.095695]
 [ 0.191419 -0.981290 -0.020690 -1.254708]
 [ 0.009467 -0.019233  0.999770  0.016659]
 [ 0.000000  0.000000  0.000000  1.000000]]
```

이 행렬은 Viser의 `Fixed GLIM spatial alignment` 폴더에서도 확인할 수 있습니다. 원본 frame을 확인하려면 `--no-glim-spatial-align`을 사용합니다.

## Same-timestamp error graph

세 trajectory를 정확히 같은 sensor timestamp grid에 보간한 뒤 다음 값을 계산합니다.

- GLIM IMU ↔ Super LIO
- GLIM corrected ↔ Super LIO
- GLIM corrected ↔ GLIM IMU
- position error: 두 position 사이의 Euclidean distance `[m]`
- orientation error: 상대 quaternion의 SO(3) angular distance `[deg]`

그래프의 굵은 노란 cursor와 6개의 현재 error marker가 선택 시각을 나타냅니다. 그래프의 line/point 클릭은 해당 시각으로 점프하고, 그래프 위에서 왼쪽 마우스로 좌우 drag하면 Viser slider, 세 robot, trajectory marker, lowstate, augmented map이 같은 timestamp로 이동합니다. 현재 오차 수치는 그래프 위의 각 pair folder에도 표시됩니다.

## Augmented map

- `bag 08-02 /augmented_map`, `bag 08-03 /augmented_map`: slider 시각보다 늦은 메시지는 사용하지 않고, 해당 시각 이전의 최신 PointCloud2를 표시합니다.
- `live /augmented_map`: ROS graph에서 실시간으로 들어오는 최신 PointCloud2를 별도 스레드로 구독합니다.
- 세 cloud는 `Augmented maps` 폴더에서 독립적으로 켜고 끌 수 있습니다.
- bag의 `/augmented_map`은 이미 Super/LIO의 `map` frame으로 기록되어 있으므로 GLIM용 `T_align`을 중복 적용하지 않습니다.
- 기본 point size는 `0.02`입니다. `Augmented maps > point size`에서 바꾼 값은 timestamp 이동으로 cloud frame이 바뀌어도 유지됩니다.

## 좌표 합성

txt pose는 `map(or odom) -> imu`로 취급합니다. robot pelvis pose는 bag의 해당 시각 TF를 이용해 아래처럼 계산합니다.

따라서 trajectory 선/축은 바닥이 아니라 로봇의 IMU 센서 높이에 보이는 것이 정상입니다. 로봇 mesh만 아래 합성으로 pelvis 위치에 배치됩니다.

```text
T_map_pelvis = T_map_imu · T_imu_lidar · T_lidar_pelvis(t)
```

URDF에 손 관절 14개가 더 있지만 bag의 `/lowstate`에는 G1 body motor 29개만 있으므로, 손 관절은 0 rad로 유지합니다.

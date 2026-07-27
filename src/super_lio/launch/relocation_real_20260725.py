import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    super_share = get_package_share_directory('super_lio')
    tools_share = get_package_share_directory('g1_localization_tools')
    del tools_share  # Package lookup intentionally fails early if tools are not built.
    config = os.path.join(
        super_share, 'config', 'relocation_real_20260725.yaml')
    rviz_config = os.path.join(
        super_share, 'rviz', 'relocation_real_20260725.rviz')

    # CMakeLists.txt is intentionally untouched, so the new map directory is
    # referenced from the workspace source tree rather than added to install().
    source_map = os.path.join(
        '/home/dongjae/Super-LIO/src/super_lio',
        'map_real_20260725', 'dump6.pcd')

    rviz = LaunchConfiguration('rviz')
    policy_bridge = LaunchConfiguration('policy_bridge')
    map_publisher = LaunchConfiguration('map_publisher')
    reference_visualizer = LaunchConfiguration('reference_visualizer')
    map_path = LaunchConfiguration('map_path')

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('policy_bridge', default_value='true'),
        DeclareLaunchArgument('map_publisher', default_value='true'),
        DeclareLaunchArgument('reference_visualizer', default_value='true'),
        DeclareLaunchArgument('map_path', default_value=source_map),
        Node(
            package='super_lio',
            executable='relocation_node',
            name='relocation_real_20260725',
            output='screen',
            parameters=[config],
            arguments=['--ros-args', '--log-level', 'info'],
        ),
        Node(
            package='g1_localization_tools',
            executable='policy_odom_bridge',
            name='policy_odom_bridge',
            output='screen',
            parameters=[{
                'publish_rate_hz': 50.0,
                'stale_timeout_sec': 0.15,
                'max_position_jump_m': 1.0,
                'max_yaw_jump_deg': 45.0,
            }],
            condition=IfCondition(policy_bridge),
        ),
        Node(
            package='g1_localization_tools',
            executable='map_publisher',
            name='prebuilt_map_publisher',
            output='screen',
            parameters=[{'map_path': map_path, 'frame_id': 'world'}],
            condition=IfCondition(map_publisher),
        ),
        Node(
            package='g1_localization_tools',
            executable='reference_visualizer',
            name='reference_visualizer',
            output='screen',
            condition=IfCondition(reference_visualizer),
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='real_bag_localization_20260725_rviz',
            output='screen',
            arguments=['-d', rviz_config, '--ros-args', '--log-level', 'warn'],
            condition=IfCondition(rviz),
        ),
    ])

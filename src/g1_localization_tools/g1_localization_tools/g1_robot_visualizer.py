import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import (
    Buffer,
    TransformBroadcaster,
    TransformException,
    TransformListener,
)
from unitree_hg.msg import LowState


G1_JOINT_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint',
    'left_hip_yaw_joint', 'left_knee_joint',
    'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint',
    'right_hip_yaw_joint', 'right_knee_joint',
    'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint', 'left_elbow_joint',
    'left_wrist_roll_joint', 'left_wrist_pitch_joint',
    'left_wrist_yaw_joint', 'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint',
    'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
]


def quaternion_multiply(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def quaternion_conjugate(quaternion):
    return (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3])


def rotate_vector(quaternion, vector):
    vector_quaternion = (vector[0], vector[1], vector[2], 0.0)
    rotated = quaternion_multiply(
        quaternion_multiply(quaternion, vector_quaternion),
        quaternion_conjugate(quaternion))
    return rotated[:3]


def normalized_quaternion(quaternion):
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1.0e-9:
        raise RuntimeError('Received a zero-length orientation quaternion')
    return tuple(value / norm for value in quaternion)


class G1RobotVisualizer(Node):
    def __init__(self):
        super().__init__('g1_recorded_motion_visualizer')
        self.declare_parameter('lowstate_topic', '/comparison/input_lowstate')
        self.declare_parameter('odometry_topic', '/comparison/input_odom')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('pelvis_frame', 'pelvis')
        self.declare_parameter('sensor_frame', 'mid360_link_IMU')
        self.declare_parameter('joint_state_stride', 10)
        self._world_frame = self.get_parameter('world_frame').value
        self._pelvis_frame = self.get_parameter('pelvis_frame').value
        self._sensor_frame = self.get_parameter('sensor_frame').value
        self._joint_state_stride = max(
            1, int(self.get_parameter('joint_state_stride').value))
        self._lowstate_count = 0
        self._base_published = False
        self._lookup_warning_emitted = False

        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=200,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._joint_publisher = self.create_publisher(
            JointState, '/joint_states', 20)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(
            LowState, self.get_parameter('lowstate_topic').value,
            self._lowstate_callback, input_qos)
        self.create_subscription(
            Odometry, self.get_parameter('odometry_topic').value,
            self._odometry_callback, input_qos)
        self.get_logger().info(
            'Publishing recorded G1 29-DOF joints and placing pelvis from '
            'Super-LIO Mid-360 pose')

    def _lowstate_callback(self, message):
        self._lowstate_count += 1
        if self._lowstate_count % self._joint_state_stride != 1:
            return
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = list(G1_JOINT_NAMES)
        count = min(len(G1_JOINT_NAMES), len(message.motor_state))
        joint_state.position = [
            float(message.motor_state[index].q) for index in range(count)]
        joint_state.velocity = [
            float(message.motor_state[index].dq) for index in range(count)]
        if count < len(G1_JOINT_NAMES):
            missing = len(G1_JOINT_NAMES) - count
            joint_state.position.extend([0.0] * missing)
            joint_state.velocity.extend([0.0] * missing)
        self._joint_publisher.publish(joint_state)

    def _odometry_callback(self, message):
        try:
            pelvis_to_sensor_message = self._tf_buffer.lookup_transform(
                self._pelvis_frame, self._sensor_frame, Time())
        except TransformException as error:
            if not self._lookup_warning_emitted:
                self._lookup_warning_emitted = True
                self.get_logger().warning(
                    f'Waiting for robot_state_publisher kinematics: {error}')
            return

        sensor_pose = message.pose.pose
        world_to_sensor_position = (
            sensor_pose.position.x, sensor_pose.position.y,
            sensor_pose.position.z)
        world_to_sensor_rotation = normalized_quaternion((
            sensor_pose.orientation.x, sensor_pose.orientation.y,
            sensor_pose.orientation.z, sensor_pose.orientation.w))
        transform = pelvis_to_sensor_message.transform
        pelvis_to_sensor_position = (
            transform.translation.x, transform.translation.y,
            transform.translation.z)
        pelvis_to_sensor_rotation = normalized_quaternion((
            transform.rotation.x, transform.rotation.y,
            transform.rotation.z, transform.rotation.w))

        sensor_to_pelvis_rotation = quaternion_conjugate(
            pelvis_to_sensor_rotation)
        sensor_to_pelvis_position = rotate_vector(
            sensor_to_pelvis_rotation,
            tuple(-value for value in pelvis_to_sensor_position))
        world_to_pelvis_rotation = normalized_quaternion(
            quaternion_multiply(
                world_to_sensor_rotation, sensor_to_pelvis_rotation))
        offset_world = rotate_vector(
            world_to_sensor_rotation, sensor_to_pelvis_position)
        world_to_pelvis_position = tuple(
            world_to_sensor_position[axis] + offset_world[axis]
            for axis in range(3))

        output = TransformStamped()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = self._world_frame
        output.child_frame_id = self._pelvis_frame
        output.transform.translation.x = world_to_pelvis_position[0]
        output.transform.translation.y = world_to_pelvis_position[1]
        output.transform.translation.z = world_to_pelvis_position[2]
        output.transform.rotation.x = world_to_pelvis_rotation[0]
        output.transform.rotation.y = world_to_pelvis_rotation[1]
        output.transform.rotation.z = world_to_pelvis_rotation[2]
        output.transform.rotation.w = world_to_pelvis_rotation[3]
        self._tf_broadcaster.sendTransform(output)

        if not self._base_published:
            self._base_published = True
            self.get_logger().info(
                'G1 model connected to map: '
                f'pelvis=({world_to_pelvis_position[0]:.3f}, '
                f'{world_to_pelvis_position[1]:.3f}, '
                f'{world_to_pelvis_position[2]:.3f})')


def main(args=None):
    rclpy.init(args=args)
    node = G1RobotVisualizer()
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

import copy
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


class ReferenceVisualizer(Node):
    def __init__(self):
        super().__init__('reference_visualizer')
        self.declare_parameter('reference_topic', '/reference/glim_odom')
        self.declare_parameter('estimated_topic', '/lio/odom')
        self.declare_parameter('world_frame', 'world')
        self._world_frame = self.get_parameter('world_frame').value
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._reference_pub = self.create_publisher(
            Odometry, '/reference/glim_odom_world', 10)
        self._path_pub = self.create_publisher(Path, '/reference/glim_path', 10)
        self._marker_pub = self.create_publisher(
            MarkerArray, '/localization/pose_markers', 10)
        self._reference_sub = self.create_subscription(
            Odometry, self.get_parameter('reference_topic').value,
            self._reference_callback, qos)
        self._estimate_sub = self.create_subscription(
            Odometry, self.get_parameter('estimated_topic').value,
            self._estimate_callback, qos)
        self._path = Path()
        self._path.header.frame_id = self._world_frame
        self._last_path_position = None
        self._reference_pose = None
        self._estimate_pose = None
        self.get_logger().info(
            'Reference adapter uses verified bag TF map->odom = identity; '
            'GLIM remains a reference trajectory, not ground truth')

    def _reference_callback(self, message):
        output = copy.deepcopy(message)
        output.header.frame_id = self._world_frame
        output.child_frame_id = 'reference_imu'
        self._reference_pub.publish(output)
        self._reference_pose = copy.deepcopy(output.pose.pose)

        position = output.pose.pose.position
        add_to_path = self._last_path_position is None
        if self._last_path_position is not None:
            distance = math.sqrt(
                (position.x - self._last_path_position[0]) ** 2 +
                (position.y - self._last_path_position[1]) ** 2 +
                (position.z - self._last_path_position[2]) ** 2)
            add_to_path = distance >= 0.05
        if add_to_path:
            pose = PoseStamped()
            pose.header = copy.deepcopy(output.header)
            pose.pose = copy.deepcopy(output.pose.pose)
            self._path.header.stamp = output.header.stamp
            self._path.poses.append(pose)
            self._path_pub.publish(self._path)
            self._last_path_position = (position.x, position.y, position.z)
        self._publish_markers(output.header.stamp)

    def _estimate_callback(self, message):
        self._estimate_pose = copy.deepcopy(message.pose.pose)
        self._publish_markers(message.header.stamp)

    def _make_marker(self, marker_id, name, pose, color, stamp):
        marker = Marker()
        marker.header.frame_id = self._world_frame
        marker.header.stamp = stamp
        marker.ns = 'localization_pose'
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = copy.deepcopy(pose)
        marker.scale.x = 0.7
        marker.scale.y = 0.12
        marker.scale.z = 0.12
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.text = name
        return marker

    def _publish_markers(self, stamp):
        markers = []
        if self._estimate_pose is not None:
            markers.append(self._make_marker(
                0, 'Super-LIO', self._estimate_pose, (0.1, 1.0, 0.1, 1.0), stamp))
        if self._reference_pose is not None:
            markers.append(self._make_marker(
                1, 'GLIM reference', self._reference_pose,
                (1.0, 0.1, 1.0, 1.0), stamp))
        if markers:
            self._marker_pub.publish(MarkerArray(markers=markers))


def main(args=None):
    rclpy.init(args=args)
    node = ReferenceVisualizer()
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

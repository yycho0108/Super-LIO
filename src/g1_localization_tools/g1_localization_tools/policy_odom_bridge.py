import copy
import math
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from .common import finite_odometry, quaternion_yaw, stamp_seconds, wrapped_angle


class PolicyOdomBridge(Node):
    def __init__(self):
        super().__init__('policy_odom_bridge')
        self.declare_parameter('input_topic', '/lio/robo/odom')
        self.declare_parameter('output_topic', '/localization/policy_odom_50hz')
        self.declare_parameter('status_topic', '/localization/status')
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('stale_timeout_sec', 0.15)
        self.declare_parameter('max_position_jump_m', 1.0)
        self.declare_parameter('max_yaw_jump_deg', 45.0)

        self._rate = float(self.get_parameter('publish_rate_hz').value)
        self._stale_timeout = float(self.get_parameter('stale_timeout_sec').value)
        self._max_position_jump = float(
            self.get_parameter('max_position_jump_m').value)
        self._max_yaw_jump = math.radians(float(
            self.get_parameter('max_yaw_jump_deg').value))

        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._publisher = self.create_publisher(
            Odometry, self.get_parameter('output_topic').value, output_qos)
        self._status_publisher = self.create_publisher(
            DiagnosticArray, self.get_parameter('status_topic').value, 10)
        self._subscription = self.create_subscription(
            Odometry, self.get_parameter('input_topic').value,
            self._source_callback, input_qos)

        self._latest = None
        self._last_source = None
        self._last_receipt_monotonic = None
        self._received = 0
        self._published = 0
        self._nonfinite_rejects = 0
        self._jump_rejects = 0
        self._out_of_order_rejects = 0
        self._stale_events = 0
        self._stale_active = False
        self._last_published_stamp = None
        self._timer = self.create_timer(1.0 / self._rate, self._publish)
        self._diagnostic_timer = self.create_timer(0.2, self._publish_status)
        self.get_logger().info(
            f'50 Hz policy bridge ready; stale timeout={self._stale_timeout:.3f}s, '
            'source timestamp is preserved')

    def _source_callback(self, message):
        self._received += 1
        if not finite_odometry(message):
            self._nonfinite_rejects += 1
            return

        if self._last_source is not None:
            stamp = stamp_seconds(message.header.stamp)
            previous_stamp = stamp_seconds(self._last_source.header.stamp)
            if stamp <= previous_stamp:
                self._out_of_order_rejects += 1
                return
            current = message.pose.pose
            previous = self._last_source.pose.pose
            distance = math.sqrt(
                (current.position.x - previous.position.x) ** 2 +
                (current.position.y - previous.position.y) ** 2 +
                (current.position.z - previous.position.z) ** 2)
            yaw_delta = abs(wrapped_angle(
                quaternion_yaw(current.orientation) -
                quaternion_yaw(previous.orientation)))
            if distance > self._max_position_jump or yaw_delta > self._max_yaw_jump:
                self._jump_rejects += 1
                return

        self._last_source = copy.deepcopy(message)
        self._latest = copy.deepcopy(message)
        self._last_receipt_monotonic = time.monotonic()
        self._stale_active = False

    def _publish(self):
        if self._latest is None or self._last_receipt_monotonic is None:
            return
        receipt_age = time.monotonic() - self._last_receipt_monotonic
        if receipt_age > self._stale_timeout:
            if not self._stale_active:
                self._stale_events += 1
                self._stale_active = True
            return
        source_stamp = stamp_seconds(self._latest.header.stamp)
        if (self._last_published_stamp is not None and
                source_stamp <= self._last_published_stamp):
            return
        output = copy.deepcopy(self._latest)
        output.child_frame_id = 'policy_imu'
        self._publisher.publish(output)
        self._last_published_stamp = source_stamp
        self._published += 1

    def _publish_status(self):
        now = self.get_clock().now().to_msg()
        message = DiagnosticArray()
        message.header.stamp = now
        status = DiagnosticStatus()
        status.name = 'g1_localization/policy_odom_bridge'
        status.hardware_id = 'super_lio'
        receipt_age = math.inf
        if self._last_receipt_monotonic is not None:
            receipt_age = time.monotonic() - self._last_receipt_monotonic

        if self._latest is None:
            status.level = DiagnosticStatus.WARN
            status.message = 'INVALID: localization not initialized'
        elif receipt_age > self._stale_timeout:
            status.level = DiagnosticStatus.ERROR
            status.message = 'INVALID: localization source is stale'
        else:
            status.level = DiagnosticStatus.OK
            status.message = 'VALID'

        values = {
            'initialized': str(self._latest is not None).lower(),
            'valid': str(status.level == DiagnosticStatus.OK).lower(),
            'receipt_age_sec': f'{receipt_age:.6f}',
            'source_timestamp_policy': 'preserved',
            'received': str(self._received),
            'published': str(self._published),
            'nonfinite_rejects': str(self._nonfinite_rejects),
            'jump_rejects': str(self._jump_rejects),
            'out_of_order_rejects': str(self._out_of_order_rejects),
            'stale_events': str(self._stale_events),
        }
        status.values = [KeyValue(key=key, value=value)
                         for key, value in values.items()]
        message.status = [status]
        self._status_publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyOdomBridge()
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

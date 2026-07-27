from pathlib import Path
import struct

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def load_xyzi_pcd(path):
    with Path(path).open('rb') as stream:
        header = {}
        while True:
            line = stream.readline()
            if not line:
                raise RuntimeError('Incomplete PCD header')
            text = line.decode('ascii').strip()
            if not text or text.startswith('#'):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == 'DATA':
                break
        if header['DATA'][0].lower() != 'binary':
            raise RuntimeError('Map publisher requires binary PCD')
        fields = header['FIELDS']
        if fields != ['x', 'y', 'z', 'intensity']:
            raise RuntimeError(f'Expected XYZI fields, got {fields}')
        if header['SIZE'] != ['4', '4', '4', '4'] or header['TYPE'] != ['F'] * 4:
            raise RuntimeError('Expected four float32 XYZI fields')
        points = int(header['POINTS'][0])
        payload = stream.read(points * 16)
        if len(payload) != points * 16:
            raise RuntimeError('PCD payload is shorter than its header')
    return points, payload


class MapPublisher(Node):
    def __init__(self):
        super().__init__('prebuilt_map_publisher')
        self.declare_parameter('map_path', '')
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('topic', '/localization/prebuilt_map')
        map_path = self.get_parameter('map_path').value
        if not map_path:
            raise RuntimeError('map_path parameter is required')
        print(F'got map_path = {map_path}')
        self._point_count, payload = load_xyzi_pcd(map_path)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._publisher = self.create_publisher(
            PointCloud2, self.get_parameter('topic').value, qos)
        self._message = PointCloud2()
        self._message.header = Header(frame_id=self.get_parameter('frame_id').value)
        self._message.height = 1
        self._message.width = self._point_count
        self._message.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12,
                       datatype=PointField.FLOAT32, count=1),
        ]
        self._message.is_bigendian = False
        self._message.point_step = 16
        self._message.row_step = 16 * self._point_count
        self._message.data = payload
        self._message.is_dense = True
        self._timer = self.create_timer(2.0, self._publish)
        self._publish()
        self.get_logger().info(
            f'Publishing transient-local prebuilt map: {map_path} '
            f'({self._point_count} points)')

    def _publish(self):
        self._message.header.stamp = self.get_clock().now().to_msg()
        self._publisher.publish(self._message)


def main(args=None):
    rclpy.init(args=args)
    node = MapPublisher()
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

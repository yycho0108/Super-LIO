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
        sizes = [int(value) for value in header['SIZE']]
        types = header['TYPE']
        counts = [int(value) for value in header.get(
            'COUNT', ['1'] * len(fields))]
        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise RuntimeError('PCD field metadata lengths do not match')
        offsets = []
        point_step = 0
        for size, count in zip(sizes, counts):
            offsets.append(point_step)
            point_step += size * count
        for required in ('x', 'y', 'z'):
            if required not in fields:
                raise RuntimeError(f'PCD is missing required field: {required}')
            index = fields.index(required)
            if sizes[index] != 4 or types[index] != 'F' or counts[index] != 1:
                raise RuntimeError(
                    f'PCD field {required} must be scalar float32')
        points = int(header['POINTS'][0])
        source_payload = stream.read(points * point_step)
        if len(source_payload) != points * point_step:
            raise RuntimeError('PCD payload is shorter than its header')

    # RViz only needs XYZ and optional intensity. Repack maps that also carry
    # normals, RGB, or other fields into a compact XYZI PointCloud2 payload.
    xyz_offsets = [offsets[fields.index(name)] for name in ('x', 'y', 'z')]
    intensity_offset = None
    if 'intensity' in fields:
        index = fields.index('intensity')
        if sizes[index] == 4 and types[index] == 'F' and counts[index] == 1:
            intensity_offset = offsets[index]

    if (fields == ['x', 'y', 'z', 'intensity'] and
            sizes == [4, 4, 4, 4] and types == ['F'] * 4 and
            counts == [1, 1, 1, 1]):
        return points, source_payload

    output = bytearray(points * 16)
    unpack_float = struct.Struct('<f')
    pack_xyzi = struct.Struct('<ffff')
    for row in range(points):
        base = row * point_step
        x, y, z = [
            unpack_float.unpack_from(source_payload, base + offset)[0]
            for offset in xyz_offsets
        ]
        intensity = (unpack_float.unpack_from(
            source_payload, base + intensity_offset)[0]
                     if intensity_offset is not None else 0.0)
        pack_xyzi.pack_into(output, row * 16, x, y, z, intensity)
    return points, bytes(output)


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

if __name__ == '__main__':
    main()

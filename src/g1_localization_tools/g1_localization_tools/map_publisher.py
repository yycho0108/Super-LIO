from pathlib import Path
import struct

import rclpy
import numpy as np
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
import open3d as o3d


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
        if not Path(map_path).exists():
            raise RuntimeError(F'map_path={map_path} does not exist')

        # Configure publisher
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._publisher = self.create_publisher(
            PointCloud2,
            self.get_parameter('topic').value,
            qos
        )


        # Read point cloud
        pcd = o3d.t.io.read_point_cloud(map_path)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        points = pcd.point.positions.numpy()

        # Optionally add intensity
        if 'intensity' in pcd.point:
            fields.append(
                PointField(name='intensity', offset=12,
                           datatype=PointField.FLOAT32, count=1),
            )
            points = np.concatenate([
                points,
                pcd.point['intensity'].numpy()
            ], axis=-1)

        # Construct message
        header = Header(frame_id=self.get_parameter('frame_id').value)
        self._message = pc2.create_cloud(
            header,
            fields,
            points.astype(np.float32)
        )

        # Periodically publish (and also, publish immediately)
        self._timer = self.create_timer(2.0, self._publish)
        self._publish()

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

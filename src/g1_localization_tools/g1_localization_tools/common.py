import math
import struct
from pathlib import Path

import numpy as np


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def quaternion_yaw(quaternion):
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


def wrapped_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def finite_odometry(message):
    pose = message.pose.pose
    twist = message.twist.twist
    values = [
        pose.position.x, pose.position.y, pose.position.z,
        pose.orientation.x, pose.orientation.y,
        pose.orientation.z, pose.orientation.w,
        twist.linear.x, twist.linear.y, twist.linear.z,
        twist.angular.x, twist.angular.y, twist.angular.z,
    ]
    if not all(math.isfinite(value) for value in values):
        return False
    norm = math.sqrt(
        pose.orientation.x ** 2 + pose.orientation.y ** 2 +
        pose.orientation.z ** 2 + pose.orientation.w ** 2)
    return 0.5 <= norm <= 1.5


def read_binary_pcd_xyz(path):
    path = Path(path)
    with path.open('rb') as stream:
        header = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f'PCD header is incomplete: {path}')
            text = line.decode('ascii').strip()
            if not text or text.startswith('#'):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == 'DATA':
                break
        if header['DATA'][0].lower() != 'binary':
            raise ValueError('Only binary PCD is supported by this helper')
        fields = header['FIELDS']
        sizes = [int(value) for value in header['SIZE']]
        types = header['TYPE']
        counts = [int(value) for value in header.get('COUNT', ['1'] * len(fields))]
        points = int(header['POINTS'][0])
        offsets = []
        offset = 0
        for size, count in zip(sizes, counts):
            offsets.append(offset)
            offset += size * count
        point_step = offset
        data = stream.read(points * point_step)
        if len(data) != points * point_step:
            raise ValueError(f'PCD payload is short: {len(data)} bytes')

    xyz = np.empty((points, 3), dtype=np.float64)
    for column, name in enumerate(('x', 'y', 'z')):
        index = fields.index(name)
        if sizes[index] != 4 or types[index] != 'F' or counts[index] != 1:
            raise ValueError(f'Unsupported PCD field representation: {name}')
        unpacker = struct.Struct('<f')
        xyz[:, column] = [
            unpacker.unpack_from(data, row * point_step + offsets[index])[0]
            for row in range(points)
        ]
    return xyz

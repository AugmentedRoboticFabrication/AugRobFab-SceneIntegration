# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# examples/python/t_reconstruction_system/integrate.py

# P.S. This example is used in documentation, so, please ensure the changes are
# synchronized.

import os
import numpy as np
import open3d as o3d
import open3d.core as o3c
import time
import matplotlib.pyplot as plt

from config import ConfigParser
from common import load_rgbd_file_names, load_depth_file_names, save_poses, load_intrinsic, load_extrinsics, get_default_testdata


def lineset_from_extrinsics(extrinsics):
    POINTS_PER_FRUSTUM = 5
    EDGES_PER_FRUSTUM = 8

    points = []
    colors = []
    lines = []

    cnt = 0
    for extrinsic in extrinsics:
        pose = np.linalg.inv(extrinsic.cpu().numpy())

        l = 0.01
        points.append((pose @ np.array([0, 0, 0, 1]).T)[:3])
        points.append((pose @ np.array([2*l, l, 2 * l, 1]).T)[:3])
        points.append((pose @ np.array([2*l, -l, 2 * l, 1]).T)[:3])
        points.append((pose @ np.array([-2*l, -l, 2 * l, 1]).T)[:3])
        points.append((pose @ np.array([-2*l, l, 2 * l, 1]).T)[:3])

        lines.append([cnt + 0, cnt + 1])
        lines.append([cnt + 0, cnt + 2])
        lines.append([cnt + 0, cnt + 3])
        lines.append([cnt + 0, cnt + 4])
        lines.append([cnt + 1, cnt + 2])
        lines.append([cnt + 2, cnt + 3])
        lines.append([cnt + 3, cnt + 4])
        lines.append([cnt + 4, cnt + 1])

        for i in range(0, EDGES_PER_FRUSTUM):
            colors.append(np.array([1, 0, 0]))

        cnt += POINTS_PER_FRUSTUM

    for i in range(len(extrinsics) - 1):
        s = i
        t = i + 1
        lines.append([POINTS_PER_FRUSTUM * s, POINTS_PER_FRUSTUM * t])
        colors.append(np.array([0, 1, 0]))

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.vstack(points))
    lineset.lines = o3d.utility.Vector2iVector(np.vstack(lines).astype(int))
    lineset.colors = o3d.utility.Vector3dVector(np.vstack(colors))

    return lineset

def integrate(depth_file_names, color_file_names, intrinsic, extrinsics,
              config):
    if os.path.exists(config.path_npz):
        print('Voxel block grid npz file {} found, trying to load...'.format(
            config.path_npz))
        vbg = o3d.t.geometry.VoxelBlockGrid.load(config.path_npz)
        print('Loading finished.')
    else:
        print('Voxel block grid npz file {} not found, trying to integrate...'.
              format(config.path_npz))

        n_files = len(depth_file_names)
        device = o3d.core.Device(config.device)

        if config.integrate_color:
            vbg = o3d.t.geometry.VoxelBlockGrid(
                attr_names=('tsdf', 'weight', 'color'),
                attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
                attr_channels=((1), (1), (3)),
                voxel_size= .006,#3.0 / 512,
                block_resolution=8,
                block_count=200000,
                device=o3d.core.Device('CUDA:0'))
        else:
            vbg = o3d.t.geometry.VoxelBlockGrid(
                attr_names=('tsdf', 'weight'),
                attr_dtypes=(o3c.float32, o3c.float32),
                attr_channels=((1), (1)),
                voxel_size= 3.0 / 512,
                block_resolution=8,
                block_count=100000,
                device=o3d.core.Device('CUDA:0'))

        start = time.time()
        for i in range(n_files):
            print('Integrating frame {}/{}'.format(i, n_files))

            depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
            extrinsic = extrinsics[i]

            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth, intrinsic, extrinsic, config.depth_scale,
                config.depth_max)

            if config.integrate_color:
                color = o3d.t.io.read_image(color_file_names[i]).to(device)
                vbg.integrate(frustum_block_coords, depth, color, intrinsic,
                              extrinsic, config.depth_scale, config.depth_max)
            else:
                vbg.integrate(frustum_block_coords, depth, intrinsic, extrinsic,
                              config.depth_scale, config.depth_max)
            
            # if i > 0:
            #     mesh = vbg.extract_triangle_mesh(-1, 0)
            #     lineset = lineset_from_extrinsics(extrinsics[:i])
            #     o3d.visualization.draw([mesh.to_legacy(), lineset])

            dt = time.time() - start
        print('Finished integrating {} frames in {} seconds'.format(
            n_files, dt))

    return vbg


if __name__ == '__main__':
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    parser.add('--integrate_color', action='store_true')
    parser.add('--path_trajectory',
               help='path to the trajectory .log or .json file.')
    parser.add('--path_npz',
               help='path to the npz file that stores voxel block grid.',
               default='vbg.npz')
    config = parser.get_config()

    if config.path_dataset == '':
        config.path_dataset = get_default_testdata()
        config.path_trajectory = os.path.join(config.path_dataset,
                                              'trajectory.log')

    if config.integrate_color:
        depth_file_names, color_file_names = load_rgbd_file_names(config)
    else:
        depth_file_names = load_depth_file_names(config)
        color_file_names = None

    intrinsic = load_intrinsic(config)
    extrinsics = load_extrinsics(config.path_trajectory, config)

    vbg = integrate(depth_file_names, color_file_names, intrinsic, extrinsics,
                    config)

    lineset = lineset_from_extrinsics(extrinsics)

    mesh = vbg.extract_triangle_mesh(-1, 0)
    o3d.io.write_triangle_mesh("./test.ply", mesh.to_legacy())
    o3d.visualization.draw([mesh.to_legacy(), lineset])


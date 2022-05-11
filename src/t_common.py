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

# examples/python/reconstruction_system/common.py

import numpy as np
import os
import open3d as o3d
import glob


def get_default_testdata():
    example_path = os.path.abspath(
        os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))

    path_dataset = os.path.join(example_path, 'test_data', 'RGBD')
    print('Dataset not found, falling back to test examples {}'.format(
        path_dataset))

    return path_dataset


def load_depth_file_names(fn, depth_folder="depth"):

    depth_folder = os.path.join(fn, depth_folder)

    # Only 16-bit png depth is supported
    depth_file_names = glob.glob(os.path.join(depth_folder, '*.png'))
    n_depth = len(depth_file_names)
    if n_depth == 0:
        print('Depth image not found in {}, abort!'.format(depth_folder))
        return []

    return sorted(depth_file_names)


def load_rgbd_file_names(fn, color_folder="color", depth_folder="depth"):
    depth_file_names = load_depth_file_names(fn)
    if len(depth_file_names) == 0:
        return [], []

    color_folder = os.path.join(fn, color_folder)
    extensions = ['*.png', '*.jpg']
    for ext in extensions:
        color_file_names = glob.glob(os.path.join(color_folder, ext))
        if len(color_file_names) == len(depth_file_names):
            return depth_file_names, sorted(color_file_names)

    depth_folder = os.path.join(fn, depth_folder)
    print('Found {} depth images in {}, but cannot find matched number of '
          'color images in {} with extensions {}, abort!'.format(
              len(depth_file_names), depth_folder, color_folder, extensions))
    return [], []


def load_intrinsic(path_intrinsic):
    intrinsic = o3d.io.read_pinhole_camera_intrinsic(path_intrinsic)
    
    return o3d.core.Tensor(intrinsic.intrinsic_matrix,
                               o3d.core.Dtype.Float64)


def load_extrinsics(path_trajectory):
    extrinsics = []
    # For either a fragment or a scene
    if path_trajectory.endswith('log'):
        data = o3d.io.read_pinhole_camera_trajectory(path_trajectory)
        for param in data.parameters:
            extrinsics.append(param.extrinsic)

    # Only for a fragment
    elif path_trajectory.endswith('json'):
        data = o3d.io.read_pose_graph(path_trajectory)
        for node in data.nodes:
            extrinsics.append(np.linalg.inv(node.pose))

    return list(map(lambda x: o3d.core.Tensor(x, o3d.core.Dtype.Float64),
                extrinsics))


def init_volume(mode, config):
    if config.engine == 'legacy':
        return o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=config.voxel_size,
            sdf_trunc=config.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    elif config.engine == 'tensor':
        if mode == 'scene':
            block_count = config.block_count
        else:
            block_count = config.block_count
        return o3d.t.geometry.TSDFVoxelGrid(
            {
                'tsdf': o3d.core.Dtype.Float32,
                'weight': o3d.core.Dtype.UInt16,
                'color': o3d.core.Dtype.UInt16
            },
            voxel_size=config.voxel_size,
            sdf_trunc=config.sdf_trunc,
            block_resolution=16,
            block_count=block_count,
            device=o3d.core.Device(config.device))
    else:
        print('Unsupported engine {}'.format(config.engine))


def extract_pointcloud(volume, config, file_name=None):
    if config.engine == 'legacy':
        mesh = volume.extract_triangle_mesh()

        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.colors = mesh.vertex_colors

        if file_name is not None:
            o3d.io.write_point_cloud(file_name, pcd)

    elif config.engine == 'tensor':
        pcd = volume.extract_point_cloud(
            weight_threshold=config.surface_weight_thr)

        if file_name is not None:
            o3d.io.write_point_cloud(file_name, pcd.to_legacy())

    return pcd


def extract_trianglemesh(volume, config, file_name=None):
    if config.engine == 'legacy':
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        if file_name is not None:
            o3d.io.write_triangle_mesh(file_name, mesh)

    elif config.engine == 'tensor':
        mesh = volume.extract_triangle_mesh(
            weight_threshold=config.surface_weight_thr)
        mesh = mesh.to_legacy()

        if file_name is not None:
            o3d.io.write_triangle_mesh(file_name, mesh)

    return mesh

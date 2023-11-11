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
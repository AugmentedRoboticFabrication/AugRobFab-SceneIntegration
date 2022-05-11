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
# import matplotlib.pyplot as plt

from .t_config import ConfigParser
from .t_common import load_rgbd_file_names, load_depth_file_names, load_intrinsic, load_extrinsics


class Integrate:
	def __init__(self, fn, root, 
				 voxel_size=3.0/512, block_count=100000, block_resolution=8, 
				 depth_max=1.0, depth_scale=1000.0, 
				 device="CUDA:0", intrinsic="intrinsic.json", trajectory="trajectory.log"):
		self.fn = fn
		self.root = root

		self.dir = os.path.join(self.root, self.fn)

		self.depth_file_names, self.color_file_names = load_rgbd_file_names(fn)
		self.intrinsic = load_intrinsic(os.path.join(self.dir, intrinsic))
		self.extrinsics = load_extrinsics(os.path.join(self.dir, trajectory))

		self.device = o3d.core.Device(device)

		self.depth_max = depth_max
		self.depth_scale = depth_scale

		self.voxel_size = voxel_size
		self.block_count = block_count
		self.block_resolution = block_resolution

		self.vbg = None

	def lineset_from_extrinsics(self):
		POINTS_PER_FRUSTUM = 5
		EDGES_PER_FRUSTUM = 8

		points = []
		colors = []
		lines = []

		cnt = 0
		for extrinsic in self.extrinsics:
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

		for i in range(len(self.extrinsics) - 1):
			s = i
			t = i + 1
			lines.append([POINTS_PER_FRUSTUM * s, POINTS_PER_FRUSTUM * t])
			colors.append(np.array([0, 1, 0]))

		lineset = o3d.geometry.LineSet()
		lineset.points = o3d.utility.Vector3dVector(np.vstack(points))
		lineset.lines = o3d.utility.Vector2iVector(np.vstack(lines).astype(int))
		lineset.colors = o3d.utility.Vector3dVector(np.vstack(colors))

		return lineset

	def integrate(self, integrate_color=True, export=True):
		n_files = len(self.depth_file_names)

		if integrate_color:
			self.vbg = o3d.t.geometry.VoxelBlockGrid(
				attr_names=('tsdf', 'weight', 'color'),
				attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
				attr_channels=((1), (1), (3)),
				voxel_size= self.voxel_size,
				block_resolution=8,
				block_count=200000,
				device=o3d.core.Device('CUDA:0'))
		else:
			self.vbg = o3d.t.geometry.VoxelBlockGrid(
				attr_names=('tsdf', 'weight'),
				attr_dtypes=(o3c.float32, o3c.float32),
				attr_channels=((1), (1)),
				voxel_size= self.voxel_size,
				block_resolution=8,
				block_count=100000,
				device=o3d.core.Device('CUDA:0'))

		start = time.time()
		for i in range(n_files):
			print('Integrating frame {}/{}'.format(i, n_files))

			depth = o3d.t.io.read_image(self.depth_file_names[i]).to(self.device)
			extrinsic = self.extrinsics[i]

			frustum_block_coords = self.vbg.compute_unique_block_coordinates(
				depth, self.intrinsic, extrinsic, self.depth_scale, self.depth_max)

			if integrate_color:
				color = o3d.t.io.read_image(self.color_file_names[i]).to(self.device)
				self.vbg.integrate(frustum_block_coords, depth, color, self.intrinsic,
							extrinsic, self.depth_scale, self.depth_max)
			else:
				self.vbg.integrate(frustum_block_coords, depth, self.intrinsic, extrinsic,
							self.depth_scale, self.depth_max)

			dt = time.time() - start
		print('Finished integrating {} frames in {} seconds'.format(
			n_files, dt))
		
		return self.vbg

	def exportMesh(self, fn="mesh.ply"):
		if self.vbg is None:
			print("No Voxel Block Grid was found, run .integrate() first!")
			return
		else:
			out_dir = os.path.join(self.dir, fn)
			print("Saving PointCloud to %s..." % out_dir, end="")
			mesh = self.vbg.extract_triangle_mesh(-1, 0)
			o3d.io.write_triangle_mesh(out_dir, mesh.to_legacy())
			print("Done!")
	
	def exportPointCloud(self, fn="pcd.ply"):
		if self.vbg is None:
			print("No Voxel Block Grid was found, run .integrate() first!")
			return
		else:
			out_dir = os.path.join(self.dir, fn)
			print("Saving PointCloud to %s..." % out_dir, end="")
			pcd = self.vbg.extract_point_cloud(-1, 0)
			o3d.io.write_point_cloud(out_dir, pcd.to_legacy())
			print("Done!")

import os
import glob

import numpy as np
import open3d as o3d
import open3d.core as o3c

from src.t_common import load_rgbd_file_names, load_depth_file_names, load_intrinsic, load_extrinsics

from src.util import readJSON, import_intrinsic_calib

class TSDF_Integration():
	"""
	A class for integrating TSDF volumes from extrinsic poses,depth and color (optional) images.

	Args:
		intrinsic_matrix: The intrinsic camera matrix of the sensor.
		voxel_size: The size of a voxel in the TSDF volume.
		block_count: The number of blocks in the TSDF volume.
		block_resolution: The resolution of each block in the TSDF volume.
		depth_max: The maximum depth cut-off used in integrating depth images.
		depth_scale: The scale factor between depth values in the depth image and the TSDF volume.
		device: The device to use for computation.
		integrate_color: Whether to integrate color information into the TSDF volume.
	"""

	def __init__(
			self,
			dir,
			intrinsic_matrix="intrinsic.json",
			voxel_size=3.0/512,
			block_count=100000,
			block_resolution=8,
			depth_max=1.0,
			depth_scale=1000.0,
			device='CUDA:0',
			depth_folder_name = 'depth',
			color_folder_name = None,
			integrate_color=False):

		self.device = o3d.core.Device(device)

		self.dir = dir
		
		self.intrinsic_matrix = self._tensorize_intrinsic_matrix(
			import_intrinsic_calib(self.dir, fn=intrinsic_matrix)[0]
			)

		self.voxel_size = voxel_size
		self.block_count = block_count
		self.block_resolution = block_resolution

		self.depth_max = depth_max
		self.depth_scale = depth_scale

		self.integrate_color = color_folder_name is not None
		self.color_folder_name = color_folder_name
		self.depth_folder_name = depth_folder_name


		self.extrinsics = []

		if self.integrate_color:
			self.vbg = o3d.t.geometry.VoxelBlockGrid(
				attr_names=('tsdf', 'weight', 'color'),
				attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
				attr_channels=((1), (1), (3)),
				voxel_size=self.voxel_size,
				block_resolution=self.block_resolution,
				block_count=self.block_count,
				device=self.device)
		else:
			self.vbg = o3d.t.geometry.VoxelBlockGrid(
				attr_names=('tsdf', 'weight'),
				attr_dtypes=(o3c.float32, o3c.float32),
				attr_channels=((1), (1)),
				voxel_size=self.voxel_size,
				block_resolution=self.block_resolution,
				block_count=self.block_count,
				device=self.device)

		self.mesh = o3d.geometry.TriangleMesh()
		self.extrinsic_lineset = o3d.geometry.LineSet()

	def integrate_frame(self, extrinsic_pose, depth_image, color_image=None):
		"""
		Integrates a new depth image and pose into the TSDF volume.

		Args:
			extrinsic_pose: The pose of the camera that captured the depth image.
			depth_image: The depth image from the camera.
			color_image: The color image from the camera (optional).
		"""
		t_extrinsic = self._tensorize_extrinsic_pose(extrinsic_pose)
		self.extrinsics.append(t_extrinsic)
		t_depth = self._tensorize_depth_image(depth_image)

		frustum_block_coords = self.vbg.compute_unique_block_coordinates(
			t_depth,
			self.intrinsic_matrix,
			t_extrinsic,
			self.depth_scale,
			self.depth_max).to(self.device)

		if self.integrate_color and color_image is not None:
			color = self._tensorize_color_image(color_image)
			self.vbg.integrate(
				frustum_block_coords,
				t_depth,
				color,
				self.intrinsic_matrix,
				t_extrinsic,
				self.depth_scale,
				self.depth_max)
		else:
			self.vbg.integrate(
				frustum_block_coords,
				t_depth,
				self.intrinsic_matrix,
				t_extrinsic,
				self.depth_scale,
				self.depth_max)
			
	def integrate_batch(self):
		if self.integrate_color:
			depth_paths, color_paths = self._load_rgbd_file_names()
		else:
			depth_paths = self._load_depth_file_names()

		trajectory_path = os.path.join(self.dir, "trajectory.log")
		extrinsic_poses = load_extrinsics(trajectory_path)

		for i in range(len(depth_paths)):
			depth = o3d.t.io.read_image(depth_paths[i]).to(self.device)
			if self.integrate_color:
				color = o3d.t.io.read_image(color_paths[i]).to(self.device)
			else:
				color = None
			print(f'Integrating {i+1}/{len(depth_paths)}...', end='')
			self.integrate_frame(
				extrinsic_pose=extrinsic_poses[i],
				depth_image=depth,
				color_image=color
				)
			print('Done!')

	def _tensorize_depth_image(self, depth_image):
		if isinstance(depth_image, o3d.cuda.pybind.t.geometry.Image):
			return depth_image.to(self.device)
		return o3d.t.geometry.Image(o3c.Tensor(depth_image)).to(self.device)

	def _tensorize_color_image(self, color_image):
		if isinstance(color_image, o3d.cuda.pybind.t.geometry.Image):
			return color_image.to(self.device)
		return o3d.t.geometry.Image(o3c.Tensor(color_image)).to(self.device)

	def _tensorize_intrinsic_matrix(self, intrinsic_matrix):
		assert (intrinsic_matrix.shape == (3, 3))

		return o3c.Tensor(intrinsic_matrix, o3c.Dtype.Float64)

	def _tensorize_extrinsic_pose(self, extrinsic_pose):
		if isinstance(extrinsic_pose, o3d.cuda.pybind.core.Tensor):
			return extrinsic_pose

		assert (extrinsic_pose.shape == (4, 4))

		return o3c.Tensor(extrinsic_pose, o3c.Dtype.Float64)

	def _load_depth_file_names(self):

		depth_folder = os.path.join(self.dir, self.depth_folder_name)

		# Only 16-bit png depth is supported
		depth_file_names = glob.glob(os.path.join(depth_folder, '*.png'))
		n_depth = len(depth_file_names)
		if n_depth == 0:
			print('Depth image not found in {}, abort!'.format(depth_folder))
			return []

		return sorted(depth_file_names)

	def _load_rgbd_file_names(self):
		depth_file_names = self._load_depth_file_names()
		if len(depth_file_names) == 0:
			return [], []

		color_folder = os.path.join(self.dir, self.color_folder_name)
		extensions = ['*.png', '*.jpg']
		for ext in extensions:
			color_file_names = glob.glob(os.path.join(color_folder, ext))
			if len(color_file_names) == len(depth_file_names):
				return depth_file_names, sorted(color_file_names)

		depth_folder = os.path.join(self.dir, self.depth_folder_name)
		print('Found {} depth images in {}, but cannot find matched number of '
			'color images in {} with extensions {}, abort!'.format(
				len(depth_file_names), depth_folder, color_folder, extensions))
		return [], []
	
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

	def extract_trianglemesh(self, file_name='mesh.ply'):
		"""
		Extracts a triangle mesh from the volume.

		Args:
			file_name (str, optional): The file name to save the mesh to. If None, the mesh will not be saved.

		Returns:
			The triangle mesh.
		"""
		mesh = self.vbg.extract_triangle_mesh()
		mesh = mesh.to_legacy()

		if file_name is not None:
			path = os.path.join(self.dir, file_name)
			o3d.io.write_triangle_mesh(file_name, mesh)
		return mesh

	def extract_pointcloud(self, file_name='pcd.ply'):
		"""
		Extracts a point cloud from the volume.

		Args:
			file_name (str, optional): The file name to save the point cloud to. If None, the point cloud will not be saved.

		Returns:
			The point cloud.
		"""
		pcd = self.vbg.extract_point_cloud()
		pcd = pcd.to_legacy()

		if file_name is not None:
			path = os.path.join(self.dir, file_name)
			o3d.io.write_point_cloud(file_name, pcd)
		return pcd

	def visualize(self):
		if not self.visualize:
			raise Exception(
				"Visualizer has not been initiated. Please use visualize = True when creating a TSDF_Integration instance.")

		self.mesh = self.extract_trianglemesh()
		self.extrinsic_lineset = self.lineset_from_extrinsics()

		o3d.visualization.draw_geometries([self.mesh, self.extrinsic_lineset])

	def lineset_from_extrinsics(self):
		"""
		Generates camera frustum lineset for visualization.
		"""
		POINTS_PER_FRUSTUM = 5
		EDGES_PER_FRUSTUM = 8

		points = []
		colors = []
		lines = []

		cnt = 0
		for extrinsic in self.extrinsics:
			pose = np.linalg.inv(extrinsic.numpy())

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
		lineset.lines = o3d.utility.Vector2iVector(
			np.vstack(lines).astype(int))
		lineset.colors = o3d.utility.Vector3dVector(np.vstack(colors))

		return lineset

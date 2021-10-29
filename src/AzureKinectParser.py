import os
import open3d as o3d
import numpy as np

import pyk4a
from pyk4a import PyK4APlayback

class AzureKinectMKVParser(object):
	def __init__(self, fn, root=None):
		self.fn = fn

		if root is None:
			self.root = os.getcwd()
		else:
			self.root = root
		
		self.mkv = '%s\\%s\\capture.mkv' %(self.root, self.fn)
		self.playback = PyK4APlayback(self.mkv)
		self.reader = o3d.io.AzureKinectMKVReader()

	def color(self):
		print('Exporting color images...', end='')
		if not os.path.exists('%s\\%s\\color' %(self.root, self.fn)):
			os.mkdir('%s\\%s\\color' %(self.root, self.fn))

		self.reader.open(self.mkv)
		index = 0
		while not self.reader.is_eof():
			rgbd = self.reader.next_frame()
			if rgbd is None:
				continue
			
			color_fn = '%s\\%s\\color\\%03d.jpg' % (self.root, self.fn, index)
			o3d.io.write_image(color_fn, rgbd.color)
			index += 1
		
		self.reader.close()

	def depth(self):
		print('Exporting depth images...', end='')
		if not os.path.exists('%s\\%s\\depth' %(self.root, self.fn)):
			os.mkdir('%s\\%s\\depth' %(self.root, self.fn))

		self.reader.open(self.mkv)
		index = 0
		while not self.reader.is_eof():
			rgbd = self.reader.next_frame()
			if rgbd is None:
				continue
			
			depth_fn = '%s\\%s\\depth\\%03d.png' % (self.root, self.fn, index)
			o3d.io.write_image(depth_fn, rgbd.depth)
			index += 1
		
		self.reader.close()

	def calibration(self):
		print('Exporting factory calibration:')
		self.playback.open()

		mtx = self.playback.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR).tolist()
		dist = self.playback.calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR).tolist()

		np.savez('%s\\%s\\calibration' %(self.root, self.fn), mtx=mtx, dist=dist)

		print('Camera Matrix:\n', mtx)
		print('Distortion Coefficients:\n',dist)

		self.playback.close()
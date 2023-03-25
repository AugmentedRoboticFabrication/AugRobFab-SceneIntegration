import os
import open3d as o3d
import cv2
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
		
		self.mkv = '%s/%s/capture.mkv' %(self.root, self.fn)
		self.playback = PyK4APlayback(self.mkv)
		self.reader = o3d.io.AzureKinectMKVReader()

		self.reader.open(self.mkv)
		metadata = self.reader.get_metadata()
		o3d.io.write_azure_kinect_mkv_metadata('%s/%s/intrinsic.json' %(self.root, self.fn), metadata)
		self.reader.close()

	def rgbd(self, depth=True, color=False, ir=False):
		self.playback.open()
		print(os.path.exists(self.fn))
		if color and not os.path.exists('%s/color' % self.fn):
			os.mkdir('%s/color' % self.fn)
		
		if depth and not os.path.exists('%s/depth' % self.fn):
			os.mkdir('%s/depth' % self.fn)

		if ir and not os.path.exists('%s/ir' % self.fn):
			os.mkdir('%s/ir' % self.fn)

		index = 0
		while True:
			try:
				capture = self.playback.get_next_capture()
				if color and capture.color is not None:
					im_color = cv2.imdecode(capture.color, cv2.IMREAD_UNCHANGED)
					im_color = cv2.cvtColor(im_color, cv2.COLOR_RGB2RGBA)

					color_fn = '%s/color/%03d.png' % ( self.fn, index)
					cv2.imwrite(color_fn, im_color)

				if depth and capture.depth is not None:
					depth_fn = '%s/depth/%03d.png' % ( self.fn, index)
					cv2.imwrite(depth_fn, capture.depth)

				if ir and capture.ir is not None:
					ir_fn = '%s/ir/%03d.png' % (self.fn, index)
					cv2.imwrite(ir_fn, capture.ir)
				index += 1
			except EOFError:
				break
		self.playback.close()
		

	def calibration(self):
		self.playback.open()

		# mtx = self.playback.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR).tolist()
		# dist = self.playback.calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR).tolist()
		# print(np.asarray(self.playback.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.DEPTH).tolist()).T.flatten())
		# np.savez('%s/%s/calibration' %(self.root, self.fn), mtx=mtx, dist=dist)

		# print('Camera Matrix:\n', mtx)
		# print('Distortion Coefficients:\n',dist)

		self.playback.close()

	def overwriteIntrinsic(self):
		import json

		self.playback.open()

		intrinsic = np.asarray(self.playback.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.DEPTH).tolist()).T.flatten()
		with open('%s/%s/intrinsic.json' % (self.root, self.fn), "r+") as f:
			data = json.load(f)
			data["intrinsic_matrix"] = intrinsic.tolist()
			capture = self.playback.get_next_capture()
			data["width"] = capture.depth.shape[0]
			data["height"] = capture.depth.shape[1]
			f.seek(0)
			json.dump(data, f)
			f.truncate()
		# print('Camera Matrix:\n', mtx)
		# print('Distortion Coefficients:\n',dist)

		self.playback.close()
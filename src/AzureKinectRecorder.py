import datetime, os, json
import open3d as o3d

class AzureKinectRecorder:
	def __init__(self, fn=None, config=None, align=False, root=None, frameMode=False):
		#global variables
		self.exit = False
		self.record = False
		self.align = align
		self.vis = o3d.visualization.VisualizerWithKeyCallback()
		self.counter = 0
		self.mode = frameMode

		#os variables
		if fn is None:
			self.fn = 'capture_{date:%Y-%m-%d-%H-%M-%S}'.format(date=datetime.datetime.now())
		else:
			self.fn = fn
		
		if root is None:
			self.root = os.getcwd()
		else:
			self.root = root
		self.abspath = '{}\\{}'.format(self.root,self.fn)
		print(self.abspath)
		
		#azure config
		if config is not None:
			self.rec_config = o3d.io.read_azure_kinect_sensor_config('{}\{}'.format(self.root,config))
		else:
			self.rec_config = o3d.io.AzureKinectSensorConfig()
		
		#init sensor
		self.recorder = o3d.io.AzureKinectRecorder(self.rec_config, 0)
		if not self.recorder.init_sensor():
			raise RuntimeError('Failed to connect to sensor!')

	def escape_callback(self):
		if self.recorder.is_record_created():
			self.recorder.close_record()
			self.record = False
		self.exit = True
		return False

	def space_callback(self):
		if not self.recorder.is_record_created():
			if not os.path.exists(self.abspath):
				os.mkdir(self.abspath)
			if self.recorder.open_record('{}/{}/capture.mkv'.format(self.root,self.fn)):
				print('Recording initialized.'
					  'Press [SPACE] capture a frame.'
					  'Press [ENTER] to save.'
					  'Press [ESC] to exit.')
				self.record = True
		if self.mode:
			print('Recording frame %03d...'%self.counter, end='')
			self.recorder.record_frame(True,self.align)
			print('Done!')
		self.counter+=1
		return False

	def run(self):
		self.vis.register_key_callback(32, self.space_callback)
		self.vis.register_key_callback(256, self.escape_callback)

		self.vis.create_window('AzureKinectRecorder', 1920, 540)

		print('Recording...'
			  'Press [SPACE] capture a frame.'
			  'Press [ESC] to save and exit.')

		geometryAdded = False
		while not self.exit:
			rgbd = self.recorder.record_frame(False, self.align)
			if rgbd is None:
				continue
			if not self.mode:
				self.recorder.record_frame(True,self.align)
			if not geometryAdded:
				self.vis.add_geometry(rgbd)
				geometryAdded = True

			self.vis.update_geometry(rgbd)
			self.vis.poll_events()
			self.vis.update_renderer()
		self.recorder.close_record()
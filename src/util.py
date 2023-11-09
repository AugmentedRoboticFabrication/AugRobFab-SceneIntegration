import numpy as np
from scipy.spatial.transform import Rotation
import os
import open3d as o3d
import glob

def composeH(R, t, inv=False):
	M = np.eye(4)
	if inv:
		R = Rotation.from_matrix(np.asarray(R)).inv().as_matrix()
		t *= -1
	M[:3, :3] = np.asarray(R)
	M[:3, 3] =  np.asarray(t).ravel()
	return M

def get_default_testdata():
    example_path = os.path.abspath(
        os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))

    path_dataset = os.path.join(example_path, 'test_data', 'RGBD')
    print('Dataset not found, falling back to test examples {}'.format(
        path_dataset))

    return path_dataset
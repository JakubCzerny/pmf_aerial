import numpy as np
import csv
import sys
from scipy.spatial import KDTree
import pdb


#Load .xyz file where point cloud is stored
with open('../dataset/odm_mesh_small_no_outliers.xyz', 'rb') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=' ')
	pointcloud = [map(float, row) for row in csvreader]


np_pointcloud = np.array(pointcloud)
kdt = KDTree(np_pointcloud[0:10000,0:2])

# np_pcloud2 = np.array(np_pointcloud[0:2].shape)
neighs = []

window_size = 5

for pt_idx, point in enumerate(np_pointcloud[0:10000,0:2]):
	print pt_idx
	pt_neighbours = kdt.query_ball_point(point, r=(window_size/2.0),p=2)

	# neighs.append(np_pointcloud[pt_neighbours])


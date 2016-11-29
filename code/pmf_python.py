import numpy as np
import csv
import sys

#Erosion function
def erosion(pointcloud, neighbours, point_idx):
	z_vals = [pointcloud[neighbour,-1] for neighbour in neighbours]
	pointcloud[point_idx, -1] = min(z_vals)
	return pointcloud

#Dilation function
def dilation(pointcloud, neighbours, point_idx):
	z_vals = [pointcloud[neighbour,-1] for neighbour in neighbours]
	pointcloud[point_idx, -1] = max(z_vals)
	return pointcloud

def morpho_filter():
	pass

#Find neighbours for a given point and window size  
def boundbox_neighbours(point, pointcloud, window_size):

	minx = point[0] - window_size/2.0
	miny = point[1] - window_size/2.0
	maxx = point[0] + window_size/2.0
	maxy = point[1] + window_size/2.0

	neighbour_inds = []
	for point_idx, p in enumerate(pointcloud):
		if (p[0]<=maxx and p[0]>=minx and p[1]<=maxy and p[1]>=miny):
			neighbour_inds.append(point_idx)
	return neighbour_inds


#Load .xyz file where point cloud is stored
with open('../dataset/odm_mesh_small_no_outliers.xyz', 'rb') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=' ')
	pointcloud = [map(float, row) for row in csvreader]

#print pointcloud
print 'size of pointcloud: ', len(pointcloud)

np_pointcloud = np.array(pointcloud)
#print np_pointcloud
#print 'type of variable: ', type(np_pointcloud)

# Build different windows and height thresholds
windows = []
height_thresholds = []

# Default parameters
slope = 1.0
cell_size = 1.0
base = 2.0
max_distance = 80
initial_distance = 13
max_window_size = 20
window_type = 'linear'

window_size = 0.0
i = 0

while window_size < max_window_size:
	# Create different windows
	if window_type == 'linear':
		window_size = cell_size * ((2*(i+1)*base) + 1)
	elif window_type == 'exponential':
		window_size = cell_size * ((2*base**(i+1) + 1))
	if window_size > max_window_size:
		break
	windows.append(window_size)
	# Create corresponding height threshold
	if i == 0:
		height_threshold = initial_distance
	elif height_threshold > max_distance:
		height_threshold = max_distance
	else:
		height_threshold = slope*(windows[-1]-windows[-2])*cell_size + initial_distance


	height_thresholds.append(height_threshold)
	i += 1

for window in windows:
	print 'window size: ', window

for thresh in height_thresholds:
	print 'height thresholds: ', thresh


flags = np.zeros(np_pointcloud.shape[0])

#Create a copy of the original file
pointcloud_copy = np.copy(np_pointcloud)

for window, thres in zip(windows, height_thresholds):

	# window = windows[-1]
	# thresh = height_thresholds[-1]


	for point_idx, point in enumerate(pointcloud_copy):
		print 'Point processed: ', point_idx

		neighbours = boundbox_neighbours(point, pointcloud_copy, window)

		if neighbours:
			# Open operator (erosion + dilation)
			pointcloud_copy = erosion(pointcloud_copy, neighbours, point_idx)
			pointcloud_copy = dilation(pointcloud_copy, neighbours, point_idx)
			# pointcloud_copy = Z




	for point_idx in range(np_pointcloud.shape[0]):
		if ( abs(np_pointcloud[point_idx,-1] - pointcloud_copy[point_idx,-1])) > thres:
			flags[point_idx] = 1
			

#Create file with non ground points
with open('../dataset/pcloud1.xyz', 'wb') as csvfile:
	csvwriter = csv.writer(csvfile, delimiter=' ')
	csvwriter.writerows(np_pointcloud[np.where(flags == 1)[0], :].tolist())

#Create file with ground points
with open('../dataset/pcloud2.xyz', 'wb') as csvfile:
	csvwriter = csv.writer(csvfile, delimiter=' ')
	csvwriter.writerows(np_pointcloud[np.where(flags == 0)[0], :].tolist())



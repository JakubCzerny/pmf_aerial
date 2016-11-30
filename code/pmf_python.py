import numpy as np
import csv
import sys
from scipy.spatial import KDTree

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
slope = 2.0
cell_size = 1.0
base = 2.0

max_distance = 100
initial_distance = 10

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

print np_pointcloud.shape
m = np.floor((np.max(np_pointcloud[:,1]) - np.min(np_pointcloud[:,1]))/cell_size) + 1
n = np.floor((np.max(np_pointcloud[:,0]) - np.min(np_pointcloud[:,0]))/cell_size) + 1


A = np.zeros((m,n,3))
print 'm, n: ', m, n

# Creating kdtree
print "Creating kdtree"
kdt = KDTree(np_pointcloud[:,0:2])
print kdt.data
print 'shape of kdtree'
print kdt.data.shape

print 'Shape of pointcloud'
print np_pointcloud.shape

min_x_value = np.min(np_pointcloud[:,0])
min_y_value = np.min(np_pointcloud[:,1])

print 'min values: '
print min_x_value, min_y_value

for row_grid_num in range(int(m)):
	print row_grid_num
	for col_grid_num in range(int(n)):
		min_x = min_x_value + row_grid_num*cell_size
		max_x = min_x + (row_grid_num+1)*cell_size
		min_y = min_y_value + col_grid_num*cell_size
		max_y = min_y + (col_grid_num+1)*cell_size


		within_x = np.where(np.logical_and(np_pointcloud[:,0]>=min_x, np_pointcloud[:,0]<=max_x))
		within_y = np.where(np.logical_and(np_pointcloud[:,1]>=min_y, np_pointcloud[:,1]<=max_y))
		selected_points = np.intersect1d(within_x, within_y)
		# print selected_points
		if len(selected_points)>0:
			# Error. You are taking the indices based on selected_points and not all the values
			p = np.argmin(np_pointcloud[selected_points,2])
			point = np_pointcloud[selected_points[p], :]
			A[row_grid_num, col_grid_num, :] = point
			# print p
		else:

			# Find clostest point to the center of the cell 
			# print 'center of cell'
			p_center = np.array([min_x+cell_size/2, min_y+cell_size/2])
			# print p_center
			nn = kdt.query(p_center, k=1)
			# print 'nn; ', nn
			interp_p = np_pointcloud[nn[1],:]
			# print 'interpolated: ', interp_p
			A[row_grid_num, col_grid_num,:] = interp_p


		# A[row_grid_num, col_grid_num,:] = np_pointcloud[selected_points,:]
		
		# print rows





sys.exit()

#Create a copy of the original file
pointcloud_copy = np.copy(np_pointcloud)

for window, thres in zip(windows, height_thresholds):

	print 'window: ', window, ' threshold: ', thres

	# window = windows[-1]
	# thresh = height_thresholds[-1]


	for point_idx, point in enumerate(pointcloud_copy):
		# print 'Point processed: ', point_idx

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



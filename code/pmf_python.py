import numpy as np
import csv
import sys
from scipy.spatial import KDTree

#Erosion function
# def erosion(pointcloud, neighbours, point_idx):
# 	z_vals = [pointcloud[neighbour,-1] for neighbour in neighbours]
# 	pointcloud[point_idx, -1] = min(z_vals)
# 	return pointcloud

def erosion(Z, window):
	Zf = np.zeros(Z.shape)
	for j in range(Z.shape[0]):
		print window

		window_range = np.unique(np.clip(np.arange(np.floor(j-window/2.0), np.floor(j+window/2.0)),0,Z.shape[0]-1))
		Zf[j] = np.min(Z[window_range.astype('int32')])
	
	return Zf

#Dilation function
# def dilation(pointcloud, neighbours, point_idx):
# 	z_vals = [pointcloud[neighbour,-1] for neighbour in neighbours]
# 	pointcloud[point_idx, -1] = max(z_vals)
# 	return pointcloud

def dilation(Z, window):
	Zf = np.zeros(Z.shape)
	for j in range(Z.shape[0]):
		print window

		window_range = np.unique(np.clip(np.arange(np.floor(j-window/2.0), np.floor(j+window/2.0)),0,Z.shape[0]-1))
		Zf[j] = np.max(Z[window_range.astype('int32')])
	
	return Zf

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
cell_size = 3
base = 2.0

max_distance = 100
initial_distance = 10

max_window_size = 30
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
flags_cells = np.zeros((m,n)) 

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


original_pts_indices = {}
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
			print 'POINTS FOUND IN CELL'
			original_pts_indices[row_grid_num*col_grid_num] = selected_points
			# Error. You are taking the indices based on selected_points and not all the values
			p = np.argmin(np_pointcloud[selected_points,2])
			point = np_pointcloud[selected_points[p], :]
			A[row_grid_num, col_grid_num, :] = point
			# print p
		else:

			# Find clostest point to the center of the cell 
			# print 'center of cell'
			print 'POINT INTERPOLATED'
			p_center = np.array([min_x+cell_size/2, min_y+cell_size/2])
			# print p_center
			nn = kdt.query(p_center, k=1)
			# print 'nn; ', nn
			interp_p = np_pointcloud[nn[1],:]
			# print 'interpolated: ', interp_p
			A[row_grid_num, col_grid_num,:] = np.array([0.0,0.0,interp_p[-1]])


		# A[row_grid_num, col_grid_num,:] = np_pointcloud[selected_points,:]
		
		# print rows


B = np.copy(A)

#Create a copy of the original file
pointcloud_copy = np.copy(np_pointcloud)

print windows
print height_thresholds

counter_flag = 0

for window, thres in zip(windows, height_thresholds):

	print 'window: ', window, ' threshold: ', thres


	# window = windows[-1]
	# thresh = height_thresholds[-1]


	for i in range(int(m)):
		# print 'Point processed: ', point_idx
		P = A[i,:,:]
		# Elevation values
		Z = P[:,2]

		Zf = erosion(Z, window)
		Zf = dilation(Zf, window)
		# print Zf
		
		
		for j in range(int(n)):
			if (abs(Z[j] - Zf[j]) > thres):
				print 'threshhhhhhhh'
				flags_cells[i,j] = window

		# sys.exit()

	for i in range(int(m)):
		for j in range(int(n)):
			if B[i,j][0] > 0 and B[i,j][1] > 0:
				if flags_cells[i,j] == 0:
					print 'THRESHHHHHHHH'
					counter_flag += 1
					# ground point
					flags[original_pts_indices[i*j]] = 1


# sys.exit()		

#Create file with non ground points
with open('../dataset/pcloud1_cell.xyz', 'wb') as csvfile:
	csvwriter = csv.writer(csvfile, delimiter=' ')
	csvwriter.writerows(np_pointcloud[np.where(flags != 0)[0], :].tolist())

#Create file with ground points
with open('../dataset/pcloud2_cell.xyz', 'wb') as csvfile:
	csvwriter = csv.writer(csvfile, delimiter=' ')
	csvwriter.writerows(np_pointcloud[np.where(flags == 0)[0], :].tolist())



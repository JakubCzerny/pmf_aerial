import csv
import numpy as np


filename = '../dataset/append_try.xyz'
with open(filename, 'rb') as pcloudfile:
	csvreader = csv.reader(pcloudfile, delimiter=' ')
	pointcloud = [map(float, row) for row in csvreader]


np_pointcloud = np.array(pointcloud)
minx = np.amin(np_pointcloud[:,0])
print minx
maxx = np.amax(np_pointcloud[:,0])
print maxx

difx = maxx - minx

with open(filename, 'ab') as pcloudfile:
	csvwriter = csv.writer(pcloudfile, delimiter=' ')
	for i in range(10):
		np_pointcloud[:,0] = np_pointcloud[:,0] + difx
		csvwriter.writerows(np_pointcloud)
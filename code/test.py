import numpy as np
import csv
import sys
import json
import sys
import time
# from scipy.spatial import cKDTree 
from scipy.spatial import cKDTree 
from scipy.spatial import kdtree
import pyspark
import cPickle


def getNeighbours(point):
    kdt_1 = kdt_b.value
    return (kdt_1.data)
#     return kdt_1.query_ball_point(point, r=(window_size/2.0),p=2)
#     kdt_1 = kdtree.KDTree(cloud_b.value[:,0:2])
#     return type(kdt_1)
#     return type(kdt_b.value)#.value.n)
#     return kdt_1.data
#     return point
#     return kdt_1.query_ball_point(point, r=(window_size/2.0),p=2)
       
    
# Build different windows and height thresholds
windows = []
height_thresholds = []

# Default parameters
slope = 1.0
cell_size = 1.0
base = 2.0
max_distance = 1
initial_distance = 0.5
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

print "========= start ========="
# file_name = "odm_mesh_small_no_outliers.xyz"
file_name = "odm_mesh_small_no_outliers2.xyz" 
# file_name = "odm_mesh_small.xyz" 
start_time = time.time()
with open(file_name, 'rb') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=' ')
    pointcloud = [map(float, row) for row in csvreader]
np_pointcloud = np.array(pointcloud)
# kdt = cKDTree(np_pointcloud[:,0:2])
kdt = cKDTree(np_pointcloud[:,0:2])


# a = cPickle.dumps(kdt)
print kdt.query([-150,-50])
a = cPickle.dumps(kdt)
b = cPickle.loads(a)
print b.query([-150,-50])

# print kdt.data
# kdt = kdtree.KDTree(np_pointcloud[:,0:2])

# kdt_b = sc.broadcast(cPickle.dumps(kdt))

# kdt_serialized = sc.serializer.dumps(kdt)






# kdt_serialized = pyspark.serializers.PickleSerializer.dumps(kdt)
cloud_b = sc.broadcast(np_pointcloud)
kdt_b = sc.broadcast(kdt)
wind_b = sc.broadcast(window_size)


kdt_u = kdt_b.value.data
print kdt_u

for pt_idx, point in enumerate(np_pointcloud[0:10000,0:2]):
#     print pt_idx
#     print type(point)
    pt_neighbours = kdt.query_ball_point(point, r=(window_size/2.0),p=2)
#     print pt_neighbours

np_indices = np.arange(len(np_pointcloud))
z_vals = [p[2] for p in np_pointcloud[np_indices]]
points = sc.textFile(file_name)
points_split = points.map(lambda x: (x.split(" ")))
points_parsed = points_split.map(lambda line: [float(line[0]), float(line[1]), float(line[2])])

print "========= erosion ========="

# getn = (lambda x: kdt.query_ball_point(x[0:2], r=(window_size/2.0),p=2))
# t =  
# t = points_parsed.
# t =  points_parsed.map(getn)
# print t.take(3)

# t = points_parsed.map(lambda x: kdt_b.value.query_ball_point(x, r=(window_size/2.0),p=2))
# t = points_parsed.map(lambda x: getNeighbours(x))
t = points_parsed.map(lambda x: getNeighbours(x))
# t = points_parsed.map(lambda x: kdt.query_ball_point([x[0],x[1]], r=50, p=2))
# print 'test point: ', kdt.query_ball_point([-255,-55], r=50, p=2)
# print kdt.data
print t.take(5)

breakdsadas

test = points_parsed.map(lambda x: ((x[0],x[1]),getNeighbours(x))).mapValues(min).map(lambda (x,y): [x[0],x[1],y])
pointcloud_eroded = np.array(test.collect())

print "========= dialation ========="

pointcloud_dialated = points_parsed.map(lambda x: ((x[0],x[1]), getNeighbours(x))).mapValues(max)#.map(lambda (x,y): [x[0],x[1],y])

window_size = windows[0]
thres = height_thresholds[0]

print "========= flags ========="
points_parsed_tuple = points_parsed.map(lambda x: ((x[0], x[1]), x[2]))
print "length points_parsed_tuple ", points_parsed_tuple.count(), "\n"
print "length pointcloud_dialated ", pointcloud_dialated.count(), "\n"

flags = points_parsed_tuple.cogroup(pointcloud_dialated)
# .map(lambda (x,y): (x,(list(y[0])[0],list(y[1])[0])) if(list(y[0])[0] != "" and list(y[1])[0] != "") else None)
# flags = flags.map(lambda (key, tuple): (key,1) if(abs(tuple[0] - tuple[1]) > thres) else (key,0))
print "========= ", time.time()-start_time ," seconds ========="
print flags.take(1)
print flags.collect()



toToFail
    
window_size = windows[0]
points_matrix = points_enum.cartesian(points_enum)
new_points = points_matrix.map(lambda (x,y): ((x[1][0], x[1][1]), y[1][2]) if ( y[1][0] >= x[1][0] - window_size/2 and y[1][0] <= x[1][0] + window_size/2 and y[1][1] >= x[1][1] - window_size/2 and y[1][1] <= x[1][1] + window_size/2 ) else None)
erosion_p = new_points.filter(lambda x: x!= None).reduceByKey(min)
erosion_p = erosion_p.map(lambda (key,z): [key[0], key[1], z])
print "\nErosion\n",erosion_p.take(5),"\n"

points_enum_2 = erosion_p.zipWithIndex().map(lambda x: (x[1],x[0]))
points_matrix_2 = points_enum_2.cartesian(points_enum_2)
new_points_2 = points_matrix_2.map(lambda (x,y): ((x[1][0], x[1][1]), y[1][2]) if ( y[1][0] >= x[1][0] - window_size/2 and y[1][0] <= x[1][0] + window_size/2 and y[1][1] >= x[1][1] - window_size/2 and y[1][1] <= x[1][1] + window_size/2 ) else None)
dialation_p = new_points_2.filter(lambda x: x!= None).reduceByKey(max)

points_parsed_tuple = points_parsed.map(lambda x: ((x[0], x[1]), x[2]))
flags = points_parsed_tuple.cogroup(dialation_p).map(lambda (x,y): (x,(list(y[0])[0],list(y[1])[0])) if(list(y[0])[0] != "" and list(y[1])[0] != "") else None)

print "\ndone\n", flags.take(15), "\n"

thres = height_thresholds[0]
flags = flags.map(lambda (key, tuple): (key,1) if(abs(tuple[0] - tuple[1]) > thres) else (key,0))

print "\n\nflags\n", flags.take(15), "\n"


# sys.exit()

# ARE WE ALLOWED TO DO THAT? IN OPEN_MP WE HAD TO TAKE CARE OF THIS
i = 0 
for window, thres in zip(windows, height_thresholds):
    print "i: ", i
    if i > 0:
        print "\n\n================ iter ================"
        window_size = windows[i]
        thres = height_thresholds[i]
        
        erosion_p = new_points.filter(lambda x: x!= None).reduceByKey(min)
        erosion_p = erosion_p.map(lambda (key,z): [key[0], key[1], z])

        points_enum_2 = erosion_p.zipWithIndex().map(lambda x: (x[1],x[0]))
        points_matrix_2 = points_enum_2.cartesian(points_enum_2)
        new_points_2 = points_matrix_2.map(lambda (x,y): ((x[1][0], x[1][1]), y[1][2]) if ( y[1][0] >= x[1][0] - window_size/2 and y[1][0] <= x[1][0] + window_size/2 and y[1][1] >= x[1][1] - window_size/2 and y[1][1] <= x[1][1] + window_size/2 ) else None)
        dialation_p = new_points_2.filter(lambda x: x!= None).reduceByKey(max)
        
        points_parsed_tuple = points_parsed.map(lambda x: ((x[0], x[1]), x[2]))
        flags_new = points_parsed_tuple.cogroup(dialation_p).map(lambda (x,y): (x,(list(y[0])[0],list(y[1])[0])) if(list(y[0])[0] != "" and list(y[1])[0] != "") else None)
        flags_new = flags_new.map(lambda (key, tuple): (key,1) if(abs(tuple[0] - tuple[1]) > thres) else (key,0))
        flags = flags_new.cogroup(flags)
        flags = flags.map(lambda (x,y): (x,(list(y[0])[0],list(y[1])[0]))).map(lambda (x,y): (x, (y[0] or y[1])))
    i+=1
print '\n\n===========finish============\n', flags.map(lambda (x,y): y).count()
    
end_time = time.time()    
print "\nExecution time: ", (end_time-start_time), "sec"
# with open('dataset/pcloud1.xyz', 'wb') as csvfile:
#     csvwriter = csv.writer(csvfile, delimiter=' ')
#     csvwriter.writerows(np_pointcloud[np.where(flags == 1)[0], :].tolist())

# with open('dataset/pcloud2.xyz', 'wb') as csvfile:
#     csvwriter = csv.writer(csvfile, delimiter=' ')
#     csvwriter.writerows(np_pointcloud[np.where(flags == 0)[0], :].tolist())


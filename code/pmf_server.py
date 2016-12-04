import numpy as np
import csv
import sys
import json
import sys
import time
from sklearn.neighbors import KDTree
import pyspark
import os
from boto.s3.connection import S3Connection



if __name__ == "__main__":
    
    #aws_access_key_id = "AKIAJXG5C3AYEYF32CKA"
    #aws_secret_access_key="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"

    #conn = S3Connection(aws_access_key_id, aws_secret_access_key) 
    #bucket = conn.get_bucket('pmf-bucket')
    #obj = bucket.get_key('odm-prueba.xyz')
    #obj.get_file(file)
    sc = pyspark.SparkContext()
    

    def getNeighbours(point):
        return np.array(kdt_b.value.query_radius(point, (window_size_b.value/2.0))[0], dtype='int32')

        
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
    file_name = "pcloudfile.xyz"
    # file_name = "odm_mesh_small_no_outliers2.xyz" 
    # file_name = "odm_mesh_small.xyz" 
    start_time = time.time()

    # print os.listdir('.')
    points = sc.textFile(sys.argv[1])#.cache()
    points_split = points.map(lambda x: (x.split(" ")))
    points_parsed = points_split.map(lambda line: [float(line[0]), float(line[1]), float(line[2])])
    np_pointcloud = np.array(points_parsed.collect())

    print "Point cloud loaded: ", time.time()-start_time
    start_time = time.time()

    kdt = KDTree(np_pointcloud[:,0:2])


    window_size = windows[0]
    thres = height_thresholds[0]
    kdt_b = sc.broadcast(kdt)
    window_size_b = sc.broadcast(window_size)


    np_indices = np.arange(len(np_pointcloud))
    z_vals = [p[2] for p in np_pointcloud[np_indices]]

    print "========= erosion ========="

    t = points_parsed.map(lambda x: ((x[0],x[1]), [point[2] for point in np_pointcloud[getNeighbours(x[0:2])]])).mapValues(min).map(lambda (x,y): [x[0], x[1], y])
    pointcloud_eroded = np.array(t.collect())

    print "========= dilation ========="

    pointcloud_dilated = points_parsed.map(lambda x: ((x[0],x[1]), [point[2] for point in pointcloud_eroded[getNeighbours(x[0:2])]])).mapValues(max)

    print "========= flags ========="
    points_parsed_tuple = points_parsed.map(lambda x: ((x[0], x[1]), x[2]))
    # print "length points_parsed_tuple ", points_parsed_tuple.count(), "\n"
    # print "length pointcloud_dilated ", pointcloud_dilated.count(), "\n"

    flags = points_parsed_tuple.cogroup(pointcloud_dilated).map(lambda (x,y): (x,(list(y[0])[0],list(y[1])[0])) if(list(y[0])[0] != "" and list(y[1])[0] != "") else None)
    flags = flags.map(lambda (key, tuple): (key,1) if(abs(tuple[0] - tuple[1]) > thres) else (key,0))
    print "========= ", time.time()-start_time ," seconds ========="
    # print flags.take(1)
    # print flags.collect()

    i = 0 
    for window, thres in zip(windows, height_thresholds):
        print "i: ", i
        if i > 0:
            print "\n\n================ iter ================"
            window_size_b = sc.broadcast(windows[i])
            thres = height_thresholds[i]
            
            print "========= erosion ========="

            t = points_parsed.map(lambda x: ((x[0],x[1]), [point[2] for point in np_pointcloud[getNeighbours(x[0:2])]])).mapValues(min).map(lambda (x,y): [x[0], x[1], y])
            pointcloud_eroded = np.array(t.collect())

            print "========= dilation ========="

            pointcloud_dilated = points_parsed.map(lambda x: ((x[0],x[1]), [point[2] for point in pointcloud_eroded[getNeighbours(x[0:2])]])).mapValues(max)

            print "========= flags ========="
            points_parsed_tuple = points_parsed.map(lambda x: ((x[0], x[1]), x[2]))
            
            print "========= flags ========="

            flags_new = points_parsed_tuple.cogroup(pointcloud_dilated).map(lambda (x,y): (x,(list(y[0])[0],list(y[1])[0])) if(list(y[0])[0] != "" and list(y[1])[0] != "") else None)
            flags_new = flags_new.map(lambda (key, tuple): (key,1) if(abs(tuple[0] - tuple[1]) > thres) else (key,0))
            flags = flags_new.cogroup(flags)
            flags = flags.map(lambda (x,y): (x,(list(y[0])[0],list(y[1])[0]))).map(lambda (x,y): (x, (y[0] or y[1])))
        i+=1
    print '\n\n===========finish============\n', 

    temp = points_parsed.map(lambda x: ((x[0],x[1]), x[2]))
    output = flags.cogroup(temp).map(lambda (x,y): ((x[0],x[1],list(y[1])[0]),list(y[0])[0]))
    non_ground = output.filter(lambda (x,y): y == 1).map(lambda (x,y): x)
    ground = output.filter(lambda (x,y): y == 0).map(lambda (x,y): x)
    # print "non_ground points number: ", output.filter(lambda (x,y): y == 1).map(lambda (x,y): x).count()
    # print "ground points number: ", output.filter(lambda (x,y): y == 0).map(lambda (x,y): x).count()
        
        
    end_time = time.time()    
    print "\nExecution time: ", (end_time-start_time), "sec"


    non_ground.saveAsTextFile(sys.argv[2])
    ground.saveAsTextFile(sys.argv[3])


#    with open('dataset/pcloud1.xyz', 'wb') as csvfile:
#        csvwriter = csv.writer(csvfile, delimiter=' ')
#        csvwriter.writerows(ground)

#    with open('dataset/pcloud2.xyz', 'wb') as csvfile:
#        csvwriter = csv.writer(csvfile, delimiter=' ')
#        csvwriter.writerows(non_ground)


import csv


new_list = []
#maxz =  
#miny = 0

with open('../dataset/odm_mesh_small.xyz', 'rb') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=' ')

	for row in csvreader:
		# print row
		if float(row[-1]) < -100:
			new_list.append(row)

# print new_list


with open('../dataset/odm_mesh_small2.xyz', 'wb') as csvfile:
	csvwriter = csv.writer(csvfile, delimiter=' ')
	csvwriter.writerows(new_list)
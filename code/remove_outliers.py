import csv


new_list = []
#maxz =  
#miny = 0

with open('odm_mesh2.xyz', 'rb') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=' ')

	for row in csvreader:
		# print row
		if float(row[-2]) > -200:
			new_list.append(row[:-1])

# print new_list


with open('odm_mesh_prueba.xyz', 'wb') as csvfile:
	csvwriter = csv.writer(csvfile, delimiter=' ')
	csvwriter.writerows(new_list)
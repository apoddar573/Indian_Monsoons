import csv
import sys
import numpy as np

f = open("bangaloreRain.csv", "rb")
data = csv.reader(f)
g = open("monsoonRainBNG.csv", "wb")
writer = csv.writer(g, delimiter = ",")
rownum = 0
for row in data:
	if rownum == 0:
		header = row
		writer.writerow(header)
	else:
		colnum = 0
		L = 0
		for col in row:
			if colnum == 0:
				year = col
				colnum += 1
			elif colnum<6:
				colnum += 1
				continue
			elif colnum>12:
				break
			elif colnum>5 and colnum<10:
				L += float(col)
				colnum += 1
		writer.writerow([year, L])
	rownum += 1


#g = open("monRainfall.csv", "wb")
"""
a = np.array([ [1, 2, 3], [4, 5, 6] ])
print a.shape
print a.ndim

b = np.arange(10)
print b

c = np.arange(3, 5)
print c

c = np.arange(2, 17, 3)
print c

c = np.linspace(2, 17, 4)
print c


c = np.linspace(2, 17, 3, endpoint = False)
print c

d = np.diag(np.arange(10))
print d
"""
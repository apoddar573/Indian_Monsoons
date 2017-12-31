import sys
import numpy as np
import csv

f = open("KamrupRain.csv", 'rb')
reader = csv.reader(f, delimiter = "\t")
g = open("monthlyKRPRain.csv", 'wb')
writer = csv.writer(g, delimiter = ",")
rownum = 0
for row in reader:
	if rownum == 1:
		header = row
		writer.writerow(['Month', 'Rain'])
	elif rownum>1:
		colnum = 0
		for col in row:
			if colnum == 0:
				year = col
			if colnum > 0:
				writer.writerow([ "%02d-%s" % (colnum, year), col])
			colnum = colnum + 1
			if colnum > 12:
				break
	rownum = rownum + 1

g.close()
f.close()

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
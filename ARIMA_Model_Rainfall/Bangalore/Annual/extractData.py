import csv
import sys
import numpy as np

f = open("bangaloreRain.csv", "rb")
data = csv.reader(f)
g = open("annualRain.csv", "wb")
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
			elif colnum>12:
				break
			else:
				L += float(col)
				colnum += 1
		writer.writerow([year, L])
	rownum += 1



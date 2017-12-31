import csv
import sys
import numpy as np

f = open("KamrupRain.csv", "rb")
data = csv.reader(f, delimiter = "\t")
g = open("annualRain.csv", "wb")
writer = csv.writer(g, delimiter = ",")
rownum = 0

for row in data:
	if rownum == 1:
		header = row
		writer.writerow(["Year", "Rain"])
	elif rownum>1:
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
				print L
				colnum += 1
		writer.writerow([year, L])
	rownum += 1



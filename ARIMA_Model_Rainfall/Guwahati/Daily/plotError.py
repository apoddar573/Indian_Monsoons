import numpy as np 
import matplotlib.pyplot as plt
import sys


def printErrors(filename):
	f = open(filename)
	z = f.read()
	f.close()

	z = z.split("]")
	z = z[1:-2]
	z[0] = z[0].replace("[", "")
	z[0] = z[0].replace(" ", "")
	z[0] = z[0].split(",")
  
	for i in range(1,len(z)):
		z[i] = z[i].replace("[", "")
		z[i] = z[i].replace(" ", "")
		z[i] = z[i][1:]
		z[i] = z[i].split(",")

	for i in range(len(z)):
		for j in range(len(z[i])):
			z[i][j] = float(z[i][j])

	print z

	f = np.array(z)
	plt.imshow(f, interpolation = "nearest", origin = "upper")
	plt.colorbar()
	plt.savefig(filename.replace("txt", "png"))
	plt.close()

	L = []
	for i in range(len(z)):
		for j in range(len(z[i])):
			L += [ [ z[i][j], (i, j) ] ]
	L = sorted(L, key = lambda x: x[0])
	print L
for i in range(2008, 2015):
	printErrors("errorsDailyGAU" + str(i) + ".txt")

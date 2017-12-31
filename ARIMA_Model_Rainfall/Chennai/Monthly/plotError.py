import numpy as np 
import matplotlib.pyplot as plt

f = open("errors.txt")
z = f.read()
f.close()

z = z.split("]")
z = z[1:-2]
print z
for i in range(len(z)):
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
plt.savefig("ErrorsBNGMonthly.png")

L = []
for i in range(len(z)):
	for j in range(len(z[i])):
		L += [ [ z[i][j], (i, j) ] ]
L = sorted(L, key = lambda x: x[0])
print L
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import cos,arccos




n = 5
tmpLst1 = []
tmpLst2 = []
tmpLst3 = []
tmpLst4 = []
while(n <= 100):
	mat1 = np.ones((n,n))
	mat2 = np.ones((n,n))
	mat3 = np.ones((n,n))
	mat4 = np.ones((n,n))


	x2 = np.zeros(n)
	x4 = np.zeros(n)

	val = 0
	while(val < n):
		x2[val] = np.cos((((2*val)+1)/(2*n))*np.pi)
		x4[val] = np.cos((((2*val)+1)/(2*n))*np.pi)
		val += 1

	i = 0
	while(i < n):
		j = 1
		while(j < n):
			mat1[i][j] = pow(np.linspace(-1,1,n)[i],j)
			mat2[i][j] = pow(x2[i],j)
			mat3[i][j] = cos(j*arccos(np.linspace(-1,1,n)[i]))
			mat4[i][j] = cos(j*arccos(x4[i]))
			j+=1
		i+=1
	n+=5
	tmpLst1.append(np.linalg.cond(mat1))
	tmpLst2.append(np.linalg.cond(mat2))
	tmpLst3.append(np.linalg.cond(mat3))
	tmpLst4.append(np.linalg.cond(mat4))

plt.semilogy(tmpLst1,label = "Equispaced nodes with monomials")
plt.semilogy(tmpLst2,label = "Chelbyshev nodes with monomials")
plt.semilogy(tmpLst3,label = "Equispaced nodes with Chelbyshev polynomials")
plt.semilogy(tmpLst4,label = "Chelbyshev nodes with Chelbyshev polynomials")
plt.legend()
plt.show()
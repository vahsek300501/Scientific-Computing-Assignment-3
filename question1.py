import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numpy.linalg as npla
import scipy.linalg as spla
import math
import pdb

def getDerivate(fx, x_val, deltaX = 1e-6):
	return (fx(x_val+deltaX)-fx(x_val-deltaX))/(2*deltaX)

def newtonMethodUtil(f,fder,x):
	return x - (f/fder)

def newtonMethod(fx,x,epochs,tolerance):

	if(abs(fx(x)) < tolerance):
		return x

	i = 0
	error_vals = []
	previous_val = x
	# while(True):
	while(i < epochs):
		fder = getDerivate(fx,x)
		if(abs(fder) < tolerance):
			break;
		f = fx(x)
		x = newtonMethodUtil(f,fder,x)
		error_vals.append(x-previous_val)
		previous_val = x
		if(abs(fx(x)) < tolerance):
			break	
		i+=1

	return x,i,error_vals

print()
print()
print("Function: x**2-1")
func = lambda x: x**2 - 1
x0 = 1000000
x,itr,error_vals = newtonMethod(func,x0,100,1e-10)
print("Calculated Values by Newton's Method: "+str(x))
ekp1 = error_vals[-1]
ek = error_vals[-2]
ekm1 = error_vals[-3]
num = math.log(ekp1/ek)
den = math.log(ek/ekm1)
print("Convergence Rate: "+str(num/den))
print()

print("Function: (x-1)**4")
func = lambda x: (x - 1)**4
x0 = 10
x,itr,error_vals = newtonMethod(func,x0,100,1e-10)
print("Calculated Values by Newton's Method: "+str(x))
ekp1 = error_vals[-1]
ek = error_vals[-2]
ekm1 = error_vals[-3]
num = math.log(ekp1/ek)
den = math.log(ek/ekm1)
print("Convergence Rate: "+str(num/den))
print()

print("Function: x-cosx")
func3= lambda x: x - math.cos(x)
x0 = 1
x,itr,error_vals = newtonMethod(func3,x0,100,1e-10)
print("Calculated Values by Newton's Method: "+str(x))
ekp1 = error_vals[-1]
ek = error_vals[-2]
ekm1 = error_vals[-3]
num = math.log(ekp1/ek)
den = math.log(ek/ekm1)
print("Convergence Rate: "+str(num/den))
print()
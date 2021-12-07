import numpy as np
from numpy import sin,cos,arccos,arctan2
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import math
import pdb

def getJacobianMatrix(r,theta,phi):
	return np.array([sin(theta)*cos(phi),r*cos(theta)*sin(phi),-1*r*sin(theta)*sin(phi),sin(theta)*sin(phi),r*cos(theta)*sin(phi),r*sin(theta)*cos(phi),cos(theta),-1*r*sin(theta),0]).reshape(3,3)

def solveFunction(x,y,z,r,theta,phi):
	return np.array([r*sin(theta)*cos(phi)-x,r*sin(theta)*sin(phi)-y,r*cos(theta)-z])

def F(x,y,z,r,theta,phi):
	return solveFunction(x, y, z, r, theta, phi)

def J(r,theta,phi):
	return getJacobianMatrix(r, theta, phi)

def newtonMethod(Fx,Jx,xPolar,xCart, epochs = 500,tolerance = 1e-12):
	
	for _ in range(0,epochs):
		tmpVal = Fx(xCart[0],xCart[1],xCart[2],xPolar[0],xPolar[1],xPolar[2])
		tmpDel = np.linalg.solve(Jx(xPolar[0],xPolar[1],xPolar[2]),-tmpVal)
		xPolar = xPolar + tmpDel
		if abs(np.linalg.norm(Fx(xCart[0],xCart[1],xCart[2],xPolar[0],xPolar[1],xPolar[2]),ord = 2)) < tolerance:
			break
	return xPolar

def getCartesianCoordinates(polar):
	return np.array([polar[0]*sin(polar[1])*cos(polar[2]),polar[0]*sin(polar[1])*sin(polar[2]),polar[0]*cos(polar[1])])

def getPolarCoordinates(cartesian):
	return np.array([math.sqrt((cartesian[0]*cartesian[0]) + (cartesian[1]*cartesian[1]) + (cartesian[2]*cartesian[2])),np.arccos(cartesian[2]/math.sqrt((cartesian[0]*cartesian[0]) + (cartesian[1]*cartesian[1]) + (cartesian[2]*cartesian[2]))),np.arctan2(cartesian[1],cartesian[0])])

def main():
	i = 0
	while(i<10):
		cartesianCoordinates = np.random.randn(1,3).reshape(3,)
		x0 = np.array([2,1,0])
		x = newtonMethod(F,J,x0,cartesianCoordinates)
		print("cartesian coordinates random generated: "+str(cartesianCoordinates))
		print("polar coordinates calculated: "+str(x))
		
		cartesianCoordinatesCalulated = getCartesianCoordinates(x)
		residualVal = np.linalg.norm(cartesianCoordinatesCalulated-cartesianCoordinates,ord = 2)/np.linalg.norm(cartesianCoordinates,ord = 2)
		print("Residual Value: "+str(residualVal))

		polarCoordinatesCalculated = getPolarCoordinates(cartesianCoordinates)
		errorVal = np.linalg.norm(x-polarCoordinatesCalculated,ord = 2)/np.linalg.norm(x,ord = 2)
		print("Error Value: "+str(errorVal))

		print()
		print()
		i+=1

main()
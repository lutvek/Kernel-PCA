import numpy as np 
import math
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import kpca

np.random.seed(6359)

"""

Implementation of methods from the paper 
Kernel PCA and De-noising in feature spaces. 
Each function has a comment above it which contains
"(e)" where e denotes the corresponding equation from 
the paper. 

Code by:
Nikos Tsardakas Renhuldt
Ludvig AAberg

"""



def genSquareData3(points, margin):
	X = []

	generatedPoints = 0
	while generatedPoints < points:
		xCord = np.random.uniform(-1.0-margin/2, 1.0+margin/2)
		yCord = np.random.uniform(-1.0-margin/2, 1.0+margin/2)
		if abs(xCord) > 1-margin/2 or abs(yCord) > 1-margin/2:
			X.append((xCord, yCord))
			generatedPoints += 1
	return np.array(X)

def genSquareData2(points, variance):
	''' returns 2-d array with data in a square '''
	points -= points%4 # safeguard
	stepSize = 2.0/(points / 4.0)
	X1 = np.arange(-1,1,stepSize) # 500 points
	X1 = np.array(zip(X1, np.ones(points/4)))
	X2 = np.arange(-1,1,stepSize) # 500 points
	X2 = np.array(zip(X2, (-np.ones(points/4))))
	X3 = np.arange(-1,1,stepSize) # 500 points
	X3 = np.array(zip(np.ones(points/4), X3))
	X4 = np.arange(-1,1,stepSize) # 500 points
	X4 = np.array(zip(-np.ones(points/4), X4))

	X = np.concatenate((X1,X2,X3,X4))

	# add noise
	for i,x in enumerate(X):
		#X[i] += np.random.multivariate_normal([0.0, 0.0], np.array([[variance, 0.0],[0.0, variance]]))
		X[i] += np.random.uniform(-variance, variance)
	return X

def genSquareData(points, variance):
	''' returns 2-d array with data in a square '''
	X = []
	for p in range(points):
		offset = np.random.uniform()
		#gaussNoise = np.random.normal(0, variance)
		uniNoise = np.random.uniform(-variance, variance)
		side = np.random.choice([-1,1])
		vertHor = np.random.choice([0,1])
		
		if vertHor:
			X.append([side+uniNoise, offset])
		else:
			X.append([offset, side+uniNoise])

	return np.array(X)

def cherryDataSet():
	''' cherry data for unit square '''
	X = [[0.05,0.777],
		[-0.025,0.775],
		[-0.325, 0.73],
		[-0.4, 0.6],
		[0.775, 0.38],
		[-0.84, 0.05],
		[0.575, -0.14],
		[-0.54, -0.22],
		[0.82, -0.225],
		[0.51, -0.29],
		[-0.74, -0.36],
		[0.69, -0.48],
		[-0.275, -0.575],
		[-0.05, -0.625],
		[-0.125, -0.626],
		[-0.576, -0.675],
		[0.62, -0.8],
		[-0.525, -0.84],
		[-0.475, -0.85]]

	return np.array(X)

def genGridData(points):
	''' returns 2-D points uniformly spread over -2 to 2 over both axis '''
	rowPoints = int(round(math.sqrt(points)))
	x = np.linspace(-2, 2, rowPoints)
	y = np.linspace(-2, 2, rowPoints)
	coords = []
	for i in range(rowPoints):
		for j in range(rowPoints):
			coords.append(np.array([x[i], y[j]]))
		
	return np.array(coords)

if __name__ == '__main__':

	# hyperparameters
	c = 1.0 # optimal for 4 components : 1.1

	# For half-circle toy example
	X, y = make_circles(n_samples=600, factor=.3, noise=.05)
	X = np.array([x for i,x in enumerate(X) if x[1]>0 and not y[i]])

	nrP = len(X)

	Xtrain = X[::int(nrP/7)]

	print len(Xtrain)
	# For square toy example

	Xtrain = genSquareData3(20, 0.03)
	#XtrainMean = np.mean(Xtrain, axis=0)
	#Xtrain -= XtrainMean
	Xtest = genGridData(20**2)
	#Xtest -= XtrainMean 


	Data = Xtrain

	l = len(Data)

	# build K
	K = kpca.createK(Data, kpca.gaussianKernel, c)

	# center K
	K = kpca.centerK(K)

	# find eigen vectors
	lLambda, alpha = np.linalg.eigh(K) # (3)
	lambdas = lLambda/l # /l with the notation from the paper (but not murphys) 
	# drop negative and 0 egienvalues and their vectors
	for i,l in enumerate(lambdas):
		if l > 10**(-8):
			lambdas = lambdas[i:]
			alpha = alpha[i:]
			break

	# use only the 4 larges eigen values with corresponding vectors
	lambdas=lambdas[-4:]
	alpha=alpha[-4:]

	# normalize alpha
	alpha = kpca.normAlpha(alpha, lambdas)

	Z =[]
	for i in range(len(Xtest)):
		Z.append(kpca.calcZ(alpha, Data, Xtest[i],kpca.gaussianKernel,c,Xtest[i]))

	# Zlin = []
	# for i in range(len(Xtest)):
	# 	Zlin.append(kpca.calcZ(alpha, Data, Xtest[i],kpca.linearKernel,c,Xtest[i]))

	Z=np.array(Z)
	#Zlin=np.array(Zlin)
	plt.plot(Xtrain.T[0], Xtrain.T[1],'ro')
	plt.plot(Z.T[0],Z.T[1],'go')
	#plt.plot(Zlin.T[0],Zlin.T[1],'bo')
	plt.show()
	
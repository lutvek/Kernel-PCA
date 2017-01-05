import numpy as np 
import math
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

def gaussianKernel(x, y, c):
	''' Returns K(x,y) where K denotes gaussian kernel '''
	return math.exp(-(np.linalg.norm(x-y)**2) / c)


def createK(data, kernelFunction, c):
	''' Returns K matrix containing inner products of the data using the kernel function 
		so that K_ij := (phi(x_i)*phi(x_j)) '''
	l = len(data)
	K = np.zeros((l,l))
	for col in range(l):
		for row in range(l):
			K[row][col] = kernelFunction(data[row],data[col], c)
	return K

def calcBetaK(alphaK, kernelFunction, data, x, c):
	''' Returns the projection of x onto the eigenvector V_k '''
	betaK = 0
	for i,xi in enumerate(data):
		betaK += alphaK[i]*kernelFunction(xi,x,c)
	return betaK	
	
def centerK(K):
	''' Returns centered K matrix, see K. Murphy 14.43 '''
	l = len(K)
	Kcentered = np.zeros((l,l))
	for i in range(l):
		for j in range(l):
			Kcentered[i][j] = K[i][j] - np.mean(K[i]) + (np.sum(K)/l**2) - np.mean(K.T[j])
	return Kcentered

def normAlpha(alpha, lambdas):
	''' Returns new alpha corresponding to normalized eigen vectors,
		so that lambda_k(a^k * a^k) = 1 '''
	for i,a in enumerate(alpha):
		a /= np.sqrt(lambdas[i])
	return alpha

def calcZ(alpha, data, x, kernelFunction, c,z0):
	''' Equation (10), returns pre-image z for single input datapoint x '''
	z = z0
	iters=0
	# calculate beta (does not change with each iteration)
	beta = [calcBetaK(aK, kernelFunction, data, x, c) for aK in alpha]

	while iters <10:
		numerator = 0
		denom = 0
		for i, xi in enumerate(data):
			#gammaI = calcGammaI(alpha, i, data, x, kernelFunction, c) * kernelFunction(z,xi,c)
			gammaI = calcGammaIOpt(alpha, i, beta) * kernelFunction(z,xi,c)
			numerator += gammaI * xi
			denom += gammaI
		if denom > 10**-12: #handling numerical instability
			z = numerator/denom
			iters +=1
		else:
			iters =0
			z=z0 + np.random.multivariate_normal(np.zeros(z0.size),np.identity(z0.size))
			numerator = 0
			denom = 0
	return z

def calcGammaI(alpha, i, data, x, kernelFunction, c):
	''' returns gamma_i = sum_{k=1}^n beta_k * alpha_i^k '''
	gammaI = 0
	alphaI = alpha.T[i]
	for k, alphaKI in enumerate(alphaI):
		gammaI += calcBetaK(alpha[k], kernelFunction, data, x, c) * alphaKI
	return gammaI

def calcGammaIOpt(alpha, i, beta):
	''' returns gamma_i = sum_{k=1}^n beta_k * alpha_i^k '''
	gammaI = 0
	alphaI = alpha.T[i]
	for k, alphaKI in enumerate(alphaI):
		gammaI += beta[k] * alphaKI
	return gammaI

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
	c = 0.2

	# For half-circle toy example
	X, y = make_circles(n_samples=600, factor=.3, noise=.05)
	X = np.array([x for i,x in enumerate(X) if x[1]>0 and not y[i]])

	nrP = len(X)

	Xtrain = X[::int(nrP/7)]

	print len(Xtrain)
	# For square toy example

	Xtrain = genSquareData2(20, 0.05)
	Xtest = genGridData(20**2) 

	Data = Xtrain

	l = len(Data)

	# build K
	K = createK(Data, gaussianKernel, c)

	# center K
	K = centerK(K)

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
	alpha = normAlpha(alpha, lambdas)

	Z =[]
	for i in range(len(Xtest)):
		Z.append(calcZ(alpha, Data, Xtest[i],gaussianKernel,c,Xtest[i]))

	Z=np.array(Z)
	plt.plot(Xtrain.T[0], Xtrain.T[1],'ro')
	plt.plot(Z.T[0],Z.T[1],'go')
	plt.show()
	
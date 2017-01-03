import numpy as np 
import math
from sklearn.datasets import make_circles

#returns K(x,y)
def gaussianKernel(x, y, c):
	return math.exp(-(np.linalg.norm(x-y)**2) / c)

def createK(data, kernelFunction, c):
	l = len(data)
	K = np.zeros((l,l))
	for col in range(l):
		for row in range(l):
			K[row][col] = kernelFunction(data[row],data[col], c)

	return K

def calcBetaK(alphaK, kernelFunction, data, x, c):
	BetaK = 0
	for i,xi in enumerate(data):
		BetaK += alphaK[i]*kernelFunction(xi,x,c)

	return BetaK	
	

def centerK(K):
	l = len(K)
	Kcentered = np.zeros((l,l))
	for i in range(l):
		for j in range(l):
			Kcentered[i][j] = K[i][j] - np.mean(K[i]) + (np.sum(K)/l**2) - np.mean(K.T[j])

	return Kcentered

def normAlpha(alpha, lambdas):
	for i,a in enumerate(alpha):
		a /= np.sqrt(lambdas[i])

	return alpha

def calcZ(alpha, data, x, kernelFunction, c,z0):
	z = z0
	iters=0
	while iters <100:
		numerator = 0
		denom = 0
		for i, xi in enumerate(data):
			numerator += calcGammaI(alpha, i, data, x, kernelFunction, c) * kernelFunction(z,xi,c)*xi
			denom += calcGammaI(alpha, i, data, x, kernelFunction, c) * kernelFunction(z,xi,c)
		z = numerator/denom
		iters +=1
	return z

def calcGammaI(alpha, i, data, x, kernelFunction, c):
	gammaI = 0
	alphaI = alpha.T[i]
	for k, alphaKI in enumerate(alphaI):
		gammaI += calcBetaK(alpha[k], kernelFunction, data, x, c) * alphaKI
	return gammaI


if __name__ == '__main__':

	# hyperparameters
	c = 0.5
	z0= np.array([0,0])

	# Data
	Data = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 5)
	l = len(Data)
	X, y = make_circles(n_samples=400, factor=.3, noise=.05)

	# build K
	K = createK(Data, gaussianKernel, c)

	# center K
	K = centerK(K)

	# find eigen vectors
	llambda, alpha = np.linalg.eigh(K)
	lambdas = llambda/l # /l with the notation from the paper (but not murphys) 
	# drop negative and 0 egienvalues and their vectors
	for i,l in enumerate(lambdas):
		if l > 10**(-8):
			lambdas = lambdas[i:]
			alpha = alpha[i:]
			break

	alpha = normAlpha(alpha, lambdas)

	print lambdas[1] * np.dot(alpha[1], alpha[1])
	print len(alpha[0])
	print len(alpha.T[0])
	print calcGammaI(alpha, 0, Data, np.array([0,1]), gaussianKernel, c)
	print calcZ(alpha, Data, np.array([1,1]), gaussianKernel, c,z0)

	# create 





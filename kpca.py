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
	BetaK = 0
	for i,xi in enumerate(data):
		BetaK += alphaK[i]*kernelFunction(xi,x,c)
	return BetaK	
	
def centerK(K):
	''' Returns centered K matrix, see K. Murphy 14.43 '''
	l = len(K)
	l_ones = np.ones((l, l), dtype=int) / l
	Kcentered = K - np.dot(l_ones,K)-np.dot(K,l_ones)+np.dot(l_ones,np.dot(K,l_ones))	
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
	while iters <5:
		numerator = 0
		denom = 0
		for i, xi in enumerate(data):
			gammaI = calcGammaI(alpha, i, data, x, kernelFunction, c) * kernelFunction(z,xi,c)
			numerator += gammaI * xi
			denom += gammaI
		z = numerator/denom
		iters +=1
	return z

def calcGammaI(alpha, i, data, x, kernelFunction, c):
	''' returns gamma_i = sum_{k=1}^n Beta_k * alpha_i^k '''
	gammaI = 0
	alphaI = alpha.T[i]
	for k, alphaKI in enumerate(alphaI):
		gammaI += calcBetaK(alpha[k], kernelFunction, data, x, c) * alphaKI
	return gammaI

if __name__ == '__main__':

	# hyperparameters
	c = 0.5
	z0= np.array([0,0])

	#For half-circle toy example
	X, y = make_circles(n_samples=600, factor=.3, noise=.05)
	X = np.array([x for i,x in enumerate(X) if x[1]>0 and not y[i]])
	Xtrain, Xtest = train_test_split(X, test_size=0.9)

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

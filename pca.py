import numpy as np 
import math
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"""
Implementation of standard PCA

Code by:
Nikos Tsardakas Renhuldt
Ludvig AAberg
"""

def vectorMult(v1,v2):
	''' returns v1*v2.T '''
	dim = len(v1)
	res = np.zeros((dim, dim))
	for i,e1 in enumerate(v1):
		for j,e2 in enumerate(v2):
			res[i][j] = e1*e2
	return res

def calcC(data):
	dim = len(data[0])
	C = np.zeros((dim,dim))
	for x in data:
		C += vectorMult(x,x)
	return C/len(data)


if __name__ == '__main__':

	# hyperparameters
	c = 0.5
	z0= np.array([0,0])

	#For half-circle toy example
	X, y = make_circles(n_samples=600, factor=.3, noise=.05)
	X = np.array([x for i,x in enumerate(X) if x[1]>0 and not y[i]])
	Xtrain, Xtest = train_test_split(X, test_size=0.95)

	# normalize test and training data
	trainMean = np.mean(Xtrain, axis=0)
	Xtrain = Xtrain - trainMean
	Xtest = Xtest - trainMean

	Data = Xtrain

	l = len(Data)
	C = calcC(Data)
	lambdas, eigenvectors = np.linalg.eigh(C)
	lambdas=lambdas[-1]
	eigenvector=eigenvectors[-1]

	Z =[]
	for x in Xtest:
		Z.append(np.dot(x,eigenvector)*eigenvector)

	Z=np.array(Z)
	plt.plot(Xtrain.T[0], Xtrain.T[1],'ro')
	plt.plot(Z.T[0],Z.T[1],'go')
	plt.show()
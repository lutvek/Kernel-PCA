import csv
import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy.io import loadmat
from skimage.util import random_noise
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA, KernelPCA


def pca_reduce(train_set, test_set, component_count):
    pca = PCA(n_components=component_count)
    pca.fit(train_set)
    return pca.inverse_transform(pca.transform(test_set))

def kpca_reduce(train_set, test_set, component_count):
    kpca = KernelPCA(kernel="rbf", n_components=component_count, fit_inverse_transform=True)
    kpca.fit(train_set)
    return kpca.inverse_transform(kpca.transform(test_set))

# Linear pca version works, rest doesn't
def figure2():
    usps = fetch_mldata('usps')
    digit = 0
    data = np.random.choice(np.where((usps.target-1) == digit)[0], size=300, replace=False)

    pca = PCA(n_components=256)
    pca.fit(usps.data[data])
    eigenvecs = pca.components_

    #for n in [1, 2, 4, 16, 32, 64, 128, 256]:
    #    plt.imshow(np.reshape(eigenvecs[n-1], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
    #    plt.show()

    kpca = KernelPCA(kernel="rbf", n_components=256, fit_inverse_transform=True)
    kpca.fit(usps.data[data])
    alphas = kpca.alphas_
    print(alphas[0])

    #for n in [1, 2, 4, 16, 32, 64, 128, 256]:
        #shape=(16,16)
        #alphas = kpca.alphas_.T[64]
        #plt.show()


# Does not work at all, the code here makes no sense atm!!
def figure3():
    usps = fetch_mldata('usps')
    digit = 3
    data = np.random.choice(np.where((usps.target-1) == digit)[0], size=350, replace=False)
    training_set, testing_set = data[:300], data[-50:]
    fractions = []
    denoised = loadmat('pre_three')['pre_images']


    for components in range(1,21):
        linear_denoised = pca_reduce(usps.data[training_set], usps.data[testing_set], components)
        linear_fraction = np.sum(np.power((denoised[components-1] - usps.data[testing_set[6]]), 2))
        plt.imshow(np.reshape(linear_denoised[0], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
        plt.show()

        print(linear_fraction)

def figure4():
    usps = fetch_mldata('usps')
    idx = np.random.randint(9298, size=3000)
    training_set = usps.data[idx,:]

    # Gaussian noise version
    for digit in range(0,10):
        data = np.random.choice(np.where((usps.target-1) == digit)[0], size=350, replace=False)
        _, testing_set = data[:300], data[-50:]
        gaussian_set = skimage.util.random_noise(usps.data[testing_set], mode='gaussian', var=0.5 ** 2)

        #Plot original testing image
        plt.subplot(13, 10, digit+1)
        plt.imshow(np.reshape(usps.data[testing_set[2]], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
        plt.axis('off')

        #Plot testing image with additive gaussian noise
        plt.subplot(13, 10, digit+11)
        plt.imshow(np.reshape(gaussian_set[2], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
        plt.axis('off')
        #plt.show()

        for i, components in enumerate([1, 4, 16, 64, 256]):
            print(components)

            linear_gaussian_digit = pca_reduce(training_set, gaussian_set, components)
            plt.subplot(13, 10, digit+21+10*i)
            plt.imshow(np.reshape(linear_gaussian_digit[2], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
            plt.axis('off')

            kernel_gaussian_digit = kpca_reduce(training_set, gaussian_set, components)
            plt.subplot(13, 10, digit+71+10*i)
            plt.imshow(np.reshape(kernel_gaussian_digit[2], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
            plt.axis('off')
    plt.show()
    
    
    # Speckle noise version
    for digit in range(0,10):
        data = np.random.choice(np.where((usps.target-1) == digit)[0], size=350, replace=False)
        _, testing_set = data[:300], data[-50:]
        snp_set = skimage.util.random_noise(usps.data[testing_set], mode='s&p', amount=0.4, salt_vs_pepper = 0.5)

        #Plot original testing image
        plt.subplot(13, 10, digit+1)
        plt.imshow(np.reshape(usps.data[testing_set[2]], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
        plt.axis('off')

        #Plot testing image with salt and pepepr noise
        plt.subplot(13, 10, digit+11)
        plt.imshow(np.reshape(snp_set[2], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
        plt.axis('off')

        for i, components in enumerate([1, 4, 16, 64, 256]):
            print(components)
            linear_snp_digit = pca_reduce(training_set, snp_set, components)
            plt.subplot(13, 10, digit+21+10*i)
            plt.imshow(np.reshape(linear_snp_digit[2], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
            plt.axis('off')

            kernel_snp_digit = kpca_reduce(training_set, snp_set, components)
            plt.subplot(13, 10, digit+71+10*i)
            plt.imshow(np.reshape(kernel_snp_digit[2], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
            plt.axis('off')
    plt.show()
    

def main():
    #figure2()
    #figure3()
    figure4()

if __name__ == "__main__":
    main()


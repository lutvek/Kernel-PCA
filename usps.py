import csv
import numpy as np
import matplotlib.pyplot as plt
import skimage
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

def figure2():
    usps = fetch_mldata('usps')
    digit = 0
    data = np.random.choice(np.where((usps.target-1) == digit)[0], size=300, replace=False)

    pca = PCA(n_components=256)
    pca.fit(usps.data[data])
    eigenvecs = pca.components_

    for n in [1, 2, 4, 16, 32, 64, 128, 256]:
        plt.imshow(np.reshape(eigenvecs[n-1], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
        plt.show()

    #kpca = KernelPCA(kernel="rbf", n_components=256, fit_inverse_transform=True)
    #kpca.fit(usps.data[data])
    #alphas = kpca.alphas_

    #for n in [1, 2, 4, 16, 32, 64, 128, 256]:
        #alphas = kpca.alphas_.T[64]
        #plt.show()


def figure3():
    usps = fetch_mldata('usps')
    digit = 3
    data = np.random.choice(np.where((usps.target-1) == digit)[0], size=350, replace=False)
    training_set, testing_set = data[:300], data[-50:]
    fractions = []

    for components in range(1,21):
        linear_denoised = pca_reduce(usps.data[training_set], usps.data[testing_set], components)
        linear_fraction = np.sum(np.power((linear_denoised[0] - usps.data[testing_set[0]]), 2))
        plt.imshow(np.reshape(linear_denoised[0], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
        plt.show()

        #print(linear_fraction)

def figure4():
    usps = fetch_mldata('usps')

    for digit in range(4,5):
        data = np.random.choice(np.where((usps.target-1) == digit)[0], size=350, replace=False)
        training_set, testing_set = data[:300], data[-50:]
        gaussian_set = skimage.util.random_noise(usps.data[testing_set], mode='gaussian', var=0.5 ** 2)
        snp_set = skimage.util.random_noise(usps.data[testing_set], mode='s&p', amount=0.4, salt_vs_pepper = 0.5)

        #Plot original testing image
        plt.imshow(np.reshape(usps.data[testing_set[0]], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
        plt.show()

        #Plot testing image with additive gaussian noise
        #plt.imshow(np.reshape(gaussian_set[0], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
        #plt.show()

        #Plot testing image with salt and pepepr noise
        plt.imshow(np.reshape(snp_set[0], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
        plt.show()

        for components in [1, 4, 16, 64, 256]:
            print(components)
            linear_snp_digit = pca_reduce(usps.data[training_set], snp_set, components)
            #plt.imshow(np.reshape(linear_snp_digit[0], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
            #plt.show()

            kernel_snp_digit = kpca_reduce(usps.data[training_set], snp_set, components)
            plt.imshow(np.reshape(kernel_snp_digit[0], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
            plt.show()

            linear_gaussian_digit = pca_reduce(usps.data[training_set], gaussian_set, components)
            #plt.imshow(np.reshape(linear_gaussian_digit[0], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
            #plt.show()

            kernel_gaussian_digit = kpca_reduce(usps.data[training_set], gaussian_set, components)
            #plt.imshow(np.reshape(kernel_gaussian_digit[0], (16, 16)), cmap=plt.cm.Greys, interpolation='none')
            #plt.show()



def main():
    #figure2()
    #figure3()
    figure4()

if __name__ == "__main__":
    main()


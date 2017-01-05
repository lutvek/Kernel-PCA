import csv
import numpy as np
import matplotlib.pyplot as plt

def get_digit_data(digit, type):
    ''' Reads digit data from file, returns each digit as a 1x256 row vector in a list '''

    if type=="train":
        num_digits = [1194, 1005, 731, 658, 652, 556, 664, 645, 542, 644]
    else:
        num_digits = [359, 264, 198, 166, 200, 160, 170, 147, 166, 177]

    digit_data = [[] for i in range(num_digits[digit])]
    with open('data/usps/' + type + str(digit) + '.csv') as f:
        i = 0
        reader = csv.reader(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONE)
        for row in reader:
            digit_data[i] = np.array(row, dtype=np.float64)
            i += 1

    return digit_data 

def sample_digits(digits, N):
    ''' Randomly sample N digits from dataset '''

    data_points = np.random.choice(range(np.shape(digits)[0]), N, replace=False)
    return np.array([digits[i] for i in data_points])

def main():
    
    digit = 0
    N_samples = 32 # Number of training samples

    data = get_digit_data(digit, "train")
    samples = sample_digits(data, N_samples)

    #print(samples)
    plt.imshow(np.reshape(samples[0], (16, 16)), cmap=plt.cm.Greys)
    plt.show()


if __name__ == "__main__":
    main()

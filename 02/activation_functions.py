import numpy as np


def sigmoid(x):
    return 1 / (1 + np.power(np.e, -x))


def sigmoidprime(x):
    return sigmoid(x) - np.power(sigmoid(x), 2)

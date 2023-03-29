import numpy as np
from scipy.special import logsumexp


# Function that check dimension compatibility between ndarrays
def assert_same_shape(array1: np.ndarray, array2: np.ndarray):
    assert array1.shape == array2.shape, 'The two arrays have incompatible shapes: {} and {}'.format(tuple(array1.shape), tuple(array2.shape))
    return None


# Function that shuffles data
def permute_data(X: np.ndarray, y: np.ndarray):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


# Function that turns a 1D array into a 2D one
def to_2d(a: np.ndarray, direction: str = "col"):
    assert a.ndim == 1, "Input tensors must be 1 dimensional"
    if direction == "col":
        return a.reshape(-1, 1)
    elif direction == "row":
        return a.reshape(1, -1)
    

# Function that computes softmax
def softmax(x, axis = None):
    return np.exp(x - logsumexp(x, axis = axis, keepdims=True))
    

# Function that normalizes an array
def normalize(a: np.ndarray):
    other = 1 - a
    return np.concatenate([a, other], axis=1)
    

# Function that un-normalizes an array
def unnormalize(a: np.ndarray):
    return a[np.newaxis, 0]
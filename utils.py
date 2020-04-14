import numpy as np

"""
ReLU function and its derivative.
"""
def relu(U):
    return np.maximum(U, 0)

def relu_prime(U):
    return (U > 0) * 1.0

"""
Sigmoid function and its derivative.
"""
def sigmoid(U):
    # A numerically-stable implementation of the sigmoid function
    return np.where(U >= 0,
                    1 / (1 + np.exp(-U)),
                    np.exp(U) / (1 + np.exp(U)))

def sigmoid_prime(U):
    Phi = sigmoid(U)
    return Phi * (1-Phi)

"""
tanh function and its derivative.
"""
def tanh(U):
    return np.tanh(U)

def tanh_prime(U):
    return 1 - np.power(np.tanh(U), 2)

"""
Softmax function. Note that the derivative is defined in Loss.CrossEntropyLoss class.
"""
def softmax(U):
    # A numerically-stable implementation of the softmax function
    exp = np.exp(U - np.max(U, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


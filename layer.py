import numpy as np
import math
import scipy.stats as stats
from param import Parameter
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, input_data, mode):
        raise NotImplementedError

    @abstractmethod
    def backward(self, delta_n):
        """
        :param delta_n: the delta from the next layer
        """
        raise NotImplementedError

class FCLayer(Layer):
    def __init__(self, input_size, output_size, initialization='normal', uniform=False):
        """
        :param input_size:      (int)   the number of nodes in the previous layer
        :param output_size:     (int)   the number of nodes in 'this' layer
        :param initialization:  (str)   whether to use Xavier initialization
        :param uniform:         (bool)  in Xavier initialization, whether to use uniform or truncated normal distribution
        """
        if initialization == 'xavier':
            # Xavier initialization followed the implementation in Tensorflow.
            fan_in = input_size
            fan_out = output_size
            n = (fan_in + fan_out) / 2.0
            if uniform:
                limit = math.sqrt(3.0 / n)
                weights = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
                bias = np.zeros((1, output_size))
            else:
                trunc_std = math.sqrt(1.3 / n)
                a, b = -2, 2                            # truncated between [- 2 * std, 2 * std]
                weights = stats.truncnorm.rvs(a, b, loc=0.0, scale=trunc_std, size=(fan_in, fan_out))
                bias = np.zeros((1, output_size))
        else:
            weights = np.random.randn(input_size, output_size)
            bias = np.random.randn(1, output_size)

        self.weights = Parameter(weights)               # instantiate Parameter object by passing the initialized values
        self.bias = Parameter(bias)                     # instantiate Parameter object by passing the initialized values
        self.param = [self.weights, self.bias]          # store the weight and bias to a list

    def forward(self, input_data, mode=False):
        """
        If self.weights.shape = (in, out), then
        :param input_data:      (numpy.ndarray, shape=[batch_size, in]) the output from the previous layer
        :param mode:            (bool) True if during training, False when testing (for Dropout)
        :return: output         (numpy.ndarray, shape=[batch_size, out])
        """
        n = input_data.shape[0]             # size of mini-batch
        self.input_data = input_data        # store the input as attribute (to use in backpropagation)

        ########## (E2) Your code goes here ##########
        output = np.dot(input_data, self.weights.value) +self.bias.value
        ##########            end           ##########
        return output

    def backward(self, delta_n):
        """
        If self.weights.shape = (in, out), then
        :param delta_n:         (numpy.ndarray, shape=[batch_size, out]) the delta from the next layer
        :return delta:          (numpy.ndarray, shape=[batch_size, in]) delta to be passed to the previous layer
        """
        ########## (E2) Your code goes here ##########
        n = delta_n.shape[0]
        delta = np.dot(delta_n , self.weights.value.T)
        dEdW = np.dot(self.input_data.T, delta_n)
        dEdb = np.dot(delta_n.T, np.ones((n,1))).T
        ##########            end           ##########

        # Store gradients
        self.weights.grad = dEdW
        self.bias.grad = dEdb
        return delta
    

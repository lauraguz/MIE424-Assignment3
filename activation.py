from layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, act, act_prime):
        """
        :param act:             (function) activation function
        :param act_prime:       (function) derivative of the activation function
        """
        self.act = act
        self.act_prime = act_prime

    def forward(self, input_data, mode=False):
        """
        :param input_data:      (numpy.ndarray, shape=[batch_size, # nodes]) output from the previous FCLayer
        :param mode:            (bool) True if during training, False when testing (for Dropout)
        :return: output:        (numpy.ndarray, shape=[batch_size, # nodes])
        """
        self.input_data = input_data

        ########## (E3) Your code goes here ##########
        output = self.act(self.input_data)
        ##########            end           ##########
        return output

    def backward(self, delta_n):
        """
        Compute and pass the delta to the previous layer
        :param delta_n:         (numpy.ndarray, shape=[batch_size, # nodes]) the delta from the next layer
        :return:                (numpy.ndarray, shape=[batch_size, # nodes]) delta to pass on to the previous layer
        """
        ########## (E3) Your code goes here ##########
        delta = delta_n * self.act_prime(self.input_data)
        ##########            end           ##########

        return delta

class Dropout(Layer):
    def __init__(self, drop_prob=0.5):
        """
        Dropout is defined as a Layer object for which forward and backward pass should be specified.
        Note that this is so-called 'inverted Dropout' which takes care of scale of outputs in the forward pass.
        :param drop_prob:       (float) probability of dropping a neuron [0, 1]
        """
        super(Dropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, input_data, mode):
        """
        Forward pass along the Dropout Layer
        :param input_data:      (numpy.ndarray) output from the previous layer
        :param mode:            (bool)          True during training and False during testing
        :return:                (numpy.ndarray)
        """
        if mode:
            self.mask = (np.random.rand(*input_data.shape) < (1 - self.drop_prob)) / (1 - self.drop_prob)
            return input_data * self.mask
        else:
            return input_data

    def backward(self, delta_n):
        return delta_n * self.mask
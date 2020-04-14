import numpy as np
from utils import softmax
from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def loss(self, pred, target):
        """
        Computes the loss function values by comparing pred and target
        :param pred:        (numpy.ndarray) the output of the output layer (after activation in MSELoss, whereas
                                            before activation in CrossEntropyLoss)
        :param target:      (numpy.ndarray) the labels
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def diff_loss(self, pred, target):
        """
        Computes the derivative of the loss function, i.e., delta
        """
        raise NotImplementedError

class MSELoss(Loss):
    def loss(self, pred, target):
        ########## (E4) Your code goes here ##########
        loss = np.mean((target - pred)**2)/2
        return loss
        ##########            end           ##########

    def diff_loss(self, pred, target):
        ########## (E4) Your code goes here ##########
        N = len(target)
        delta =(1/N)*(pred - target)
        return delta
        ##########            end           ##########


class CrossEntropyLoss(Loss):
    """
    This class combines the cross entropy loss with the softmax outputs as done in the PyTorch implementation.
    The return value of self.diff_loss will then be directly handed over to the output FCLayer
    """
    def loss(self, pred, target):
        pred = softmax(pred)
        ########## (E4) Your code goes here ##########

        loss=   -1* np.sum(target* np.log(pred))
        return loss
        ##########            end           ##########

    def diff_loss(self, pred, target):
        ########## (E4) Your code goes here ##########
        pred = softmax(pred)
        delta = pred - target
        return delta
        ##########            end           ##########


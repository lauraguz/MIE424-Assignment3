#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 22:14:39 2020

@author: lauraguzman
"""


from model import NeuralNetwork
import numpy as np
import unittest

from layer import FCLayer
in_dim, out_dim, num_sample = 2, 4, 10
x = np.ones((num_sample, in_dim))

nn = NeuralNetwork()
nn.add(FCLayer(in_dim, out_dim))
layer = nn.layers[0]

pred = nn.predict(x)
delta_n = np.random.randn(num_sample, out_dim)
delta = layer.backward(delta_n)


assert pred.shape == (num_sample, out_dim), "Failed method: 'forward'\tIncorrect output shape."
assert delta.shape == (num_sample, in_dim), "Failed method: 'backward\tShape mismatch"
assert layer.weights.grad.shape == layer.weights.shape, "Failed method: 'backward'\tIncorrect dEdW shape"
assert layer.bias.grad.shape == layer.bias.shape, "Failed method: 'backward'\tIncorrect dEdb shape"

from utils import sigmoid_prime, sigmoid
from activation import Activation
in_dim, hidden_dim, out_dim, num_sample = 2, 8, 4, 10
x = np.ones((num_sample, in_dim))

nn = NeuralNetwork()
nn.add(FCLayer(in_dim, hidden_dim))
nn.add(Activation(sigmoid, sigmoid_prime))
nn.add(FCLayer(hidden_dim, out_dim))
sigmoid_layer = nn.layers[1]

nn.predict(x)
act = nn.layers[-1].input_data
delta_n = np.random.randn(num_sample, out_dim)
delta_n = nn.layers[-1].backward(delta_n)
delta = sigmoid_layer.backward(delta_n)

assert act.shape == (num_sample, hidden_dim), "Failed method: 'forward'\tIncorrect hidden layer shape."
assert delta.shape == (num_sample, hidden_dim), "Failed method: 'backward\tShape mismatch"

from loss import MSELoss
num_samples = 10
pred = np.ones((num_samples, 1))
target = np.zeros((num_samples, 1))
loss = MSELoss()
mse = loss.loss(pred, target)
mse_diff = loss.diff_loss(pred, target)

assert mse.size == 1, "Failed method: 'loss' in MSELoss\tIncorrect loss shape."
assert mse_diff.shape == pred.shape, "Failed method: 'diff_loss' in MSELoss\tIncorrect diff_loss shape"

from loss import CrossEntropyLoss
out_dim = 10
pred = np.random.rand(num_samples, out_dim)
target = np.eye(num_samples)
ce_loss = CrossEntropyLoss()
ce = ce_loss.loss(pred, target)
ce_diff = ce_loss.diff_loss(pred, target)

assert ce.size == 1, "Failed method: 'loss' in CrossEntropyLoss\tIncorrect loss shape"
assert ce_diff.shape == pred.shape, "Failed method: 'diff_loss' in CrossEntropyLoss\tIncorrect diff_loss shape"
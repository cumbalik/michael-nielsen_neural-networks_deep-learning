# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 19:48:43 2017

@author: Cingis Alexander
"""

import mnist_loader
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()

training_data = list(training_data)
np.shape(training_data[0][0])

net = Network([784, 30, 10])
net.SGD(training_data, 30,10,3.0, test_data=test_data)


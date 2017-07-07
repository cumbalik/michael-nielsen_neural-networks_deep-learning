# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 19:48:43 2017

@author: Cingis Alexander
Figure out which what is the structure of  training_data, validation_data and
test_data. Write it precisely down to use the data from InFactory
"""
import numpy as np
import mnist_loader
import time

training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()

#!!! You have to pass the SGD function list object, therefore you haveto cast it
test_data = list(test_data)
training_data = list(training_data)
validation_data = list(validation_data)
np.shape(training_data[0][0])



net = Network([784, 100, 10])
#start = time.time()
net.SGD(training_data, 30,10,0.3, 
        lmbda = 5,
        evaluation_data = validation_data,
        monitor_evaluation_accuracy = True
        
        )
#end = time.time()
#print(end-start)

net1 = Network([784, 10])
net1.SGD(training_data, 30, 20, 3.0, test_data=test_data)



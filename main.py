# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 19:48:43 2017

@author: Cingis Alexander
"""

import mnist_loader
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()

test_data = list(test_data)
training_data = list(training_data)
np.shape(training_data)

net = Network([784, 30, 10])
net.SGD(training_data, 30,10,3.0, test_data=test_data)

net1 = Network([784, 10])
net1.SGD(training_data, 30, 20, 3.0, test_data=test_data)

import numpy as np
mini_batch = training_data[15:50]
np.shape(training_data[0][0])
X = np.empty(np.shape(training_data[0][0]))
Y = np.empty(np.shape(training_data[0][1]))
for x,y in mini_batch:
    X = np.append(X,x, axis=1)
    Y = np.append(Y,y, axis=1)
    
X = np.asarray(X)    
X = X[:,1:]
print(np.array_equal(X[:,0].reshape(784,1),training_data[15][0]))
for bool in (X[:,0]==training_data[15][0])
np.shape(X)
X = np.array(X)
X[0][0][0]
np.asarray(mini_batch[0][0])

delta =np.asarray( [[1,2],[3,4]])
np.sum(delta,axis=1).reshape(np.shape(delta)[0],1)


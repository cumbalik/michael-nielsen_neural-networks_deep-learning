# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 21:44:42 2017

@author: Cingis Alexander
"""

import numpy as np
import theano.tensor as T
from theano import function

#Adding two scalars
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y],z)

f(2,3)
np.allclose(f(16.3, 12.1), 28.4)

#Adding two matrices
x = T.dmatrix('x')
y = T.dmatrix('y')

#Computing more than one thing at
#the same time
a, b = T.matrices('a','b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
f = function([a,b],[diff, abs_diff, diff_squared])

from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=234)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))

f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)    #Not updating rv_n.rng
f_val0 = f()
f()
g()

#Shared Variables
from theano import shared
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates = [(state, state+inc)])
print(state.get_value())
accumulator(1)

#Nice thin is you can define more than one function
#use the same shared variable
decrementor = function([inc], state, updates = [(state, state - inc)])
decrementor(1)
state.get_value()

fn_of_state = state * 2 + inc
foo = T.scalar(dtype = state.dtype)
skip_shared = function([inc], fn_of_state)
skip_shared(1)
state.get_value()

#Copying functions the advantage, the compilation has to be done only once
new_state =  shared(0)
new_accumulator = accumulator.copy(swap = {state: new_state})
new_accumulator(100)

null_accumulator = accumulator.copy(delete_updates = True)
null_accumulator(10)


#Randomnes
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
srng = RandomStreams(seed = 234)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates= True)

print(f()) #If we call this function 2x we get different values
print(g()) #If we call this function 2x we get same values


#Real example: Logistic Regression
import numpy as np
import theano
import theano.tensor as T
rng = np.random

N = 400 #training sample size
feats = 784 #number of input variables

#generate a dataset: D = (input_values, target_class)
D = (rng.rand(N, feats), rng.randint(size = N, low = 0, high = 2))
training_steps = 10000

#Declare Theano symbolic variables
x = T.dmatrix("x")
y = T.dvector("y")

#initialize the weight vector w randomly
#this and the following bias variable b 
#are shared so they keep their values
#between training iterations (updates)
w = theano.shared(rng.randn(feats), name = "w")
b = theano.shared(0., name = "b")

print("Initial model:")
print(w.get_value())
print(b.get_value())

# Construct Theano expression graph
p_1 = 1/(1 + T.exp(-T.dot(x, w) - b))
prediction = p_1 > 0.5
xent = -y * T.log(p_1) - (1-y) * T.log(1 - p_1)
cost = xent.mean() + 0.01*(w**2).sum()
gw, gb = T.grad(cost, [w, b]) #Compute the gradient of the cost w.r.t. weight
                                # vector w and bias term

# Compile 
train = theano.function(inputs = [x,y],
                        outputs = [prediction, xent],
                        updates = ((w, w - 0.1 * gw),(b, b - 0.1 * gb)))
predict = theano.function(inputs = [x], outputs = prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])
    
print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))

np.sum(abs(D[1] - predict(D[0])))
#%% Derivatives in Theano, it pretty cool!!! <3
import numpy as np
import theano
import theano.tensor as T
from theano import pp
x = T.dvector('x')
y = x**2
gy = theano.gradient.jacobian(y,x)
pp(gy)
f = theano.function([x], gy)
f(4)
np.allclose(f(94.2),188.4)

x = T.dmatrix('x')
s = T.sum(1 / (1 + T.exp(-x)))
gs = T.grad(s, x)
dlogistic = theano.function([x], gs)
dlogistic([[0, 1], [-1, -2]])

x = np.array([3,4,5,6,2])
x[T.arange(3)]


a = theano.shared(np.asarray([[1,2],[3,4]]), 'a')
c = a.reshape((-1,1))
print(c.shape)
c.eval()


















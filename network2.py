# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 22:19:19 2017

@author: Cingis Alexander

~~~~~~~~~
"""

#Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np

#Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):
    
    @staticmethod
    def fn(a,y):
        return 0.5*np.linalg.norm(a-y)**2
    
    @staticmethod
    def delta(z,a,y):
        return (a-y)*simgoid_prime(z)
    
class CrossEntropyCost(object):
    
    @staticmethod
    def fn(a,y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    @staticmethod
    def delta(z,a,y):
        return (a-y)


#### Main Network Class
class Network(object):
    
    def __init__(self, sizes, cost = CrossEntropyCost):
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
        
    def default_weight_initializer(self):
        
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for
                        x,y in zip(self.sizes[:-1],self.sizes[1:])]
        
    def large_weight_initializer(self):
        
        self.biases= [np.random.rand(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x) for 
                        x,y in zip(self.sizes[:-1],self.sizes[1:])]
        
    def feedforward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
            
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data = None,
            monitor_evaluation_cost =  False,
            monitor_evaluation_accuracy = False,
            monitor_training_cost = False,
            monitor_training_accuracy = False,
            early_stopping_n = 0):
        """ Train the neuaral network using mini-batch stochastic gradient
        descent. The 'training_data' is a list of tuples (x,y) representing
        the training inputs and the desired outputs. The other non-optional
        parameters are self-explanatory, as is the regularization parameter
        'lmbda'"""        
        
        # early stopping functionality:
        best_accuracy = 0
        no_accuracy_change = 0
                
        training_data = list(training_data)
        n = len(training_data)
        
        if evaluation_data:
#            print("Here?")
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)
#            print(evaluation_data)
            
        evaluation_cost, evaluation_accuracy = [],[]
        training_cost, training_accuracy = [],[]
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size]
                            for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch_vec(mini_batch, eta, lmbda, n)
                
            print("Epoch %s training complete" %j)
    
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
                
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert = True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {}/{}".format(accuracy,n))
                
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert = True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
                
            if monitor_evaluation_accuracy:
              
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {}/{}".format(accuracy,
                      n_data))
                
            #Early stopping
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                    print("Early-stopping: Best so far {}".format(best_accuracy))
                else:
                    no_accuracy_change +=1
                    
                if (no_accuracy_change == early_stopping):
                    print("Early-stopping: No accuracy change in last epoch: {}"
                          .format(early_stopping))
                    return evaluation_cost, evaluation_accuracy, training_cost,\
                            training_accuracy
        return evaluation_cost, evaluation_accuracy, training_cost,\
                            training_accuracy
        
        
    def update_mini_batch_vec(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        nabla_b, nabla_w = self.backprop_vec(mini_batch)
        tmp = len(mini_batch)
        self.biases = [b - (eta/tmp)* nb for b,nb in zip(self.biases,nabla_b)]
        self.weights = [(1-eta*(lmbda/n))*w -(eta/tmp)*nw for w,nw in zip(self.weights, nabla_w)]
    
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """ Update the network's weights and biases by applying gradient descent
        using backpropagation to singe mini batch"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w,nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb
                            for b,nb in zip(self.biases, nabla_b)]
    
    def backprop_vec(self, mini_batch):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #The numpy array is going to contain in each column a learning example
        #Y the coresponding target 
        X= np.empty(np.shape(mini_batch[0][0]))
        Y = np.empty(np.shape(mini_batch[0][1]))
        for x,y in mini_batch:
            X =np.append(X,x, axis = 1)
            Y = np.append(Y,y, axis = 1)
        X = X[:,1:]; Y = Y[:,1:]
        
        #We will now calculate all training examples "paralel" in feedforward
        #phase
        activations = X
        activations_list = [X]
        z_list = []
        for w,b in zip(self.weights, self.biases):
            Z = np.dot(w,activations) + b
            z_list.append(Z)
            activations = sigmoid(Z)
            activations_list.append(activations)
            
        delta = self.cost.delta(z_list[-1],activations_list[-1],Y)
        nabla_b[-1] = np.sum(delta,axis = 1).reshape(np.shape(delta)[0],1)
        
        summy = 0
        for i in range(len(mini_batch)):
            delta_tmp = delta[:,i]
            activation_tmp = activations_list[-2][:,i]
            if delta_tmp.ndim == 1:
               delta_tmp = delta_tmp.reshape(np.shape(delta)[0],1)
            if activation_tmp.ndim == 1:
                activation_tmp =activation_tmp.reshape(1,np.shape(activation_tmp)[0])
                
            summy +=np.dot(delta_tmp, activation_tmp)
            
        nabla_w[-1] = summy
        for l in range(2,self.num_layers):
            Z = z_list[-l]
            sp = sigmoid_prime(Z)
            delta = np.dot(self.weights[-l+1].transpose(),delta)*sp
            nabla_b[-l] = np.sum(delta, axis = 1).reshape(np.shape(delta)[0],1)
            summa = 0
            for i in range(len(mini_batch)):
                delta_tmp = delta[:,i]
                activation_tmp = activations_list[-l-1][:,i].transpose()
                if delta_tmp.ndim == 1:
                    delta_tmp = delta_tmp.reshape(np.shape(delta_tmp)[0],1)
                if activation_tmp.ndim == 1:
                    activation_tmp = activation_tmp.reshape(1,np.shape(activation_tmp)[0])
                    
                summa = summa + np.dot(delta_tmp, activation_tmp)
                
            nabla_w[-l] = summa
            
        return nabla_b,nabla_w
        
        
        
                

           
    def backprop(self, x, y):
        """ Return a tupel (nabla_b, nabla_w) representing  the gradient for
        the cost function C_x """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        #feedforward
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        #backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
        return (nabla_b, nabla_w)
    
    def accuracy(self, data, convert = False):
        if convert:
            results = [(np.argmax(self.feedforward(x)),np.argmax(y))
                        for x,y in data]
        else:
            results = [(np.argmax(self.feedforward(x)),y)
                        for x,y in data]
            
        result_accuracy = sum(x==y for x,y in results)
        return result_accuracy
    
    def total_cost(self, data, lmbda, convert = False):
        cost = 0
        for x,y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a,y)/len(data)
            cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)

        return cost
    
    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
        
#### Loading a Network
def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost = cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
        
        
#### Miscellaneous functions
def vectorized_result(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

"""That's the case where no vectorization is used
start = time.time()
net.SGD(training_data, 20,5,0.3, 
        lmbda = 10,
        evaluation_data=test_data,
        monitor_evaluation_accuracy = True
        
        )
end = time.time()
107.61370921134949
Now the case where vectorization is used
91.45837807655334

"""













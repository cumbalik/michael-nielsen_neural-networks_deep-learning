"""
network.py
~~~~~~~~~~
A module to implement the stochastic gradient descent learning
algorithm for feedforward neural network. Gradients are calculated 
using backpropagatin. Note that I have focused on making the code simple,
easily readable, and easily modifiable. It is not optmized, and ommits
many desirable features.

All variables must be declared as array i.e. vector of dim 1xn musst be 
an array of shape [1,n]

"""

###Libraries
# Standard library 
import random

# Third-party libraries
import numpy as np

class Network(object):
    
    def __init__(self,size):
        """
        The list 'size' contains the number of neurons in the 
        respective layers of the network. For exampla, if the list
        was [2,3,1] then it would be a three-layer network, with the first
        layer containing 2 neurons, the second layer containing three neurons
        and the third layer containing 1 neuron. The biases and weights for
        the network are inialized randomly, using a Gaussian distribution with
        mean 0 and variance 1. Note that the first layer is assumed to be an
        input  layer, and by convention we won't set any biases for those 
        neurons, since biases are only ever used in computing the outputs from
        later layers
        """
        self.num_layers = len(size)
        self.size = size
        self.biases = [np.random.randn(y,1) for y in size[1:]]
        self.weights = [np.random.randn(y,x)
                        for x,y in zip(size[:-1], size[1:])]
        """
        size[:-1] means we run from beginn of the elem in size till the before-
        last elem in size
        zip("Foo","Tua") outputs [[F,T][o,u],[o,a]]
        """
        
    def feedforward(self, a):
        """Return the output of the network if 'a' is input """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
            
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the data usinig mini-batch stochastic gradient descent.
        The 'training_data' is a list of tuples '(x,y)' representing the training
        inputs and the desired outputs.
        """
        training_data = list(training_data)
        if test_data:
            print("HUrra")
            test_data = list(test_data)
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                    training_data[k:k+mini_batch_size]
                    for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                
            if test_data:
                print("Epochs{0}:{1} / {2}".format(
                        j, self.evaluate(test_data), n_test))
                
            else:
                print("Epoch {0} complete".format(j))
        
    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying gradient descent
        using backpropagataion to a single mini batch. The 'mini_batch' is a
        list of tuples '(x,y)', and 'eta' is the learning rate
        The structure of mini_batch= [(["the vector x1"], y), ([..],y),...]
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [ nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [ nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w,nw in zip(self.weights, nabla_w) ]
        self.biases = [b- (eta/len(mini_batch))*nb
                       for b,nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y):
        """Return a tuple '(nabla_b, nabla_w)' representing the gradient
        for the cost function C_x. 'nabla_b' and 'nabla_w'are layer-by-layer
        lists of numpy arrays, similar to 'self.biases' and 'self.weights'
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the vectors,layer by layer
        for b, w in zip(self.biases, self.weights):
            #print("shape of b",np.shape(b))
            #print("shape of w",np.shape(w))
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # backward pass
        delta = self.cost_derivative(activations[-1],y)*\
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        #Note that the variable l in the loop below is used a liitle differently
        #to the notation in chapter 2 of the book. Here, l = 1 means the last
        #layer of neurons, l=2 is the second-layer, and so on. 
        
        for l in range(2,self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural network outputs
        the corret results. Note that the neural network's output is assumed to
        be the index of whichever neuron in the final layer has the highest activation
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
    
    def cost_derivative(self, output_activations,y):
        """Return the vector of partial derivatives for the output
        activations."""
        return (output_activations-y)
    
    ### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    out = 1.0/(1.0+np.exp(-z))
    return out

def sigmoid_prime(z):
    """Derivative of sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
    
#    import mnist_loader
#    import imp
#    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#    imp.reload(mnist_loader)    
        
            
            
            
            
            
            
            
        
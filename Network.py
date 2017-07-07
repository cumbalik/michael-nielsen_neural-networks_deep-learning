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
#        for b in self.biases:
#            print("b shape", np.shape(b), b.shape)
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
                self.update_mini_batch_vec(mini_batch, eta)

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
        for b in self.biases:
            print("b shape", np.shape(b))
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
    
    def update_mini_batch_vec(self, mini_batch, eta):
        """ The gets a mini_batch and hands the backprop_vec the whole mini_batch
        over instead one learning example and its target """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
#        for b in self.biases:
#            print("update minibatch vor b shape", np.shape(b))
        
        nabla_b, nabla_w = self.backprop_vec(mini_batch)
        
#        for b in self.biases:
#            print("update minibach nach b shape", np.shape(b))
        """ The following code calls serial two loops instead you can call just one
        loop and within an iteration you can do both calculations"""
        self.weights = [w - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb
                       for b,nb in zip(self.biases, nabla_b)]
#        for b in self.biases:
#            print("update minibach noch nach b shape", np.shape(b))
        
    def backprop_vec(self, mini_batch):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        """We need first to store all learning examples and targets as numpy
        array. The array X contains for each column a learning example and Y the
        same"""
        X = np.empty(np.shape(mini_batch[0][0]))
        Y = np.empty(np.shape(mini_batch[0][1]))
        for x,y in mini_batch:
            X = np.append(X,x, axis=1)
            Y = np.append(Y,y,axis=1)
#        X = np.asarray(X);Y=np.asarray(Y)
        X = X[:,1:]; Y = Y[:,1:]
#        print("Shape of X:",X.shape, np.shape(X))
#        print("Shape of Y:",Y.shape, np.shape(Y))
        #You can check the equality:
#        n1 = np.shape(mini_batch[0][0])[0]
#        n2 = np.shape(mini_batch[0][1])[0]
#        print("Number of true should be:",len(mini_batch))
#        for i in range(0,len(mini_batch),1):
#            print(np.array_equal(Y[:,i].reshape(n2,1),mini_batch[i][1]))
        #The X and Y are correct:)
        
        """ We will now calculate in "parallel" all training examples """
        # feedforward
        activations  = X
        activations_list = [X]
        z_list = []
#        ct=0
#        for b in self.biases:
#            print("b shape", np.shape(b), b.shape)
        """ In backprop function would be this loop calld for each traning
        example in mini_batch"""
        for w,b in zip(self.weights, self.biases):
            #b = b.transpose()
            #b = np.array([b,]*len(mini_batch)).transpose()
#            print("shape of b in",ct,np.shape(b))
            Z = np.dot(w, activations) +b
#            act0 = activations[:,4].reshape(np.shape(activations[:,4])[0],1)
#            z1 = np.dot(w,act0) +b
#            print("shpae of z1",np.shape(z1), "shape fof b", np.shape(b))
#            print("shpae of activa",np.shape(activations[:,4]))
#            print("printing t",z1,Z[:,4])
#            print("shape of Z with no b:",ct,np.shape(Z))
            z_list.append(Z)
            activations = sigmoid(Z)
#            act1 = sigmoid(z1)
#            print("printing act1",act1,activations[:,4])
            activations_list.append(activations)
            
        #print(activations_list[0].shape, activations_list[1].shape,activations_list[2].shape)
        """ In backward pass is the calculation idea the same as in update_mini_batch
        by the determination of nabla_b and nabla_w i.e. a just moved the two for-loops
        into here """
        # backward pass
        delta = self.cost_derivative(activations_list[-1],Y)*sigmoid_prime(z_list[-1])
        #sum([delta[:,i] for i in range(len(mini_batch))])
        nabla_b[-1] = np.sum(delta, axis=1).reshape(np.shape(delta)[0],1)
        summy=0
        for i in range(len(mini_batch)):
            delta_tmp = delta[:,i]
            activation_tmp = activations_list[-2][:,i]
            if delta_tmp.ndim == 1:
                delta_tmp = delta_tmp.reshape(np.shape(delta_tmp)[0],1)
            if activation_tmp.ndim == 1:
                activation_tmp = activation_tmp.reshape(1,np.shape(activation_tmp)[0])
                    
            summy = summy +np.dot(delta_tmp, activation_tmp)
            
        nabla_w[-1] = summy
        
#        nabla_w[-1] = sum([np.dot(delta[:,i].reshape(np.shape(delta)[0],1),\
#                          (activations_list[-2][:,i].reshape(activations_list[-2][:,i][0],1)).transpose())\
#                          for i in range(len(mini_batch))])
        # we backpropagate the error and calculate the gradient
        for l in range(2, self.num_layers):
            Z = z_list[-l]
            sp = sigmoid_prime(Z)
            delta =np.dot(self.weights[-l+1].transpose(),delta)*sp
            #[delta[:,i] for i in range(len(mini_batch))]
            nabla_b[-l] = np.sum(delta, axis =1).reshape(np.shape(delta)[0],1)
            """The problem occurs in the last part of layer
            ValueError: shapes (30,) and (784,) not aligned: 30 (dim 0) != 784 (dim 0)
            so you have to have (30,1) amd (784,1).
            The dimension of a with shape (30,) is a.ndim is if such case occurs
            change it to (30,1)
            """
#            print(np.shape(activations_list[-l-1][:,0]))
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
#            nabla_w[-l] = sum([np.dot(delta[:,i],activations_list[-l-1][:,i].transpose())\
#                          for i in range(len(mini_batch))])
            
        
        return (nabla_b,nabla_w)        
    
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
    
            
            
            
            
            
            
            
        
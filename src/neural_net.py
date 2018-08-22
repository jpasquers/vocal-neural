import numpy as np
import random
from src.weights import Weights
from src.sample import Sample
from src.iteration import Iteration
import sys

class NeuralNet():
    
    def __init__(self, layer_lengths, learning_rate=.3, lmbda=.1, numIter=200,epsilon=0.125):
        """
        layer_lengths will be an array where each value is the size of that layer
        e.x. [3,4,5] - would have an input layer of size 3, a hidden layer of size 4, 
        and an output layer of size 5.
        Note: These do not include bias terms.
        """
        #initialize weights
        self.weights = Weights(layer_lengths, epsilon)
        self.layer_lengths = layer_lengths
        self.num_layers = len(self.layer_lengths)
        self.learning_rate = learning_rate
        #index of the last layer
        self.L = self.num_layers - 1
        self.lmbda = lmbda

    def train(self,X,Y,num_iterations):
        """
        where m = X.shape[1] = Y.shape[1] = #samples
        X = self.layer_lengths[0] x m ndarray
        Y = self.layer_lengths[self.L] x m ndarray
        """
        if X.shape[1] != Y.shape[1]:
            #TODO proper error checking
            print("inequal sample sizes for input and outputs")
            sys.exit()
        for i in range(0,num_iterations):
            print("starting iteration: " + str(i))
            self.train_iteration(X,Y)

    def train_iteration(self,X,Y):
        iteration = Iteration(self.weights,X,Y,self.lmbda)
        partials = iteration.calc_error_partials()

        #TODO some function that takes in theta and dJ/dTheta and gives the new theta
        #for now, do a simple change of dJ/dtheta * learning_rate. Eventually convert
        #to a more sophisticated gradient descent algorithm
        for i in range(0,self.L):
            next_theta = np.subtract(self.weights.get_layer(i), self.learning_rate * partials[i])
            self.weights.update_layer(i, next_theta)
    def error(self, expected, actual):
        """
        .5 * Sum(i) (expected[i] - actual[i])^2
        """
        diff = np.subtract(expected, actual)
        diff_squared = np.square(diff)
        return 0.5 * np.sum(diff_squared)




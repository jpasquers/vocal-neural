import numpy as np
import random
from scipy.special import expit
import sys

class NeuralNet():
    
    def __init__(self, layer_lengths, learning_rate=.2, lmbda=.1, numIter=200,epsilon=0.125):
        """
        layer_lengths will be an array where each value is the size of that layer
        e.x. [3,4,5] - would have an input layer of size 3, a hidden layer of size 4, 
        and an output layer of size 5.
        Note: These do not include bias terms.
        """
        self.EPSILON = epsilon
        self.layer_lengths = layer_lengths
        self.num_layers = len(self.layer_lengths)
        #index of the last layer
        self.L = self.num_layers - 1
        self.lmbda = lmbda
        self.reset_thetas()
        self.reset_per_iteration()

    def reset_per_sample(self):
        self.reset_alphas()
        self.reset_zs()
        self.reset_ds()
        self.reset_deltas()

    def reset_per_iteration(self):
        self.reset_per_sample()
        self.reset_Ds()
    
    def reset_alphas(self):
        """
        Use an array of numpy arrays to allow us to vary layer_lengths.
        First value will always be a bias term except for in the last layer
        """
        self.alphas = []
        for i, layer_length in enumerate(self.layer_lengths):
            #The last column will have no bias term
            if i == self.L:
                self.alphas.append(np.zeros([layer_length,1]))
            else:
                self.alphas.append(np.zeros([layer_length+1,1]))

    def reset_zs(self):
        """
        Use an array of numpy arrays to allow us to vary layer_lengths.
        Note: The first column of z will never be used.
        """
        self.zs = [np.zeros([layer_length,1]) for layer_length in self.layer_lengths]

    def get_rand(self):
        """
        Calculate a random value off of the given EPSILON value in the range [-self.EPSILON, self.EPSILON]
        """
        return random.random() * 2 * self.EPSILON - self.EPSILON

    def reset_thetas(self):
        """
        Theta[0] will map from a[0] to z[1]. Therefore, if layer 0 has size m, and layer 1 has size n,
        Then Theta[0] will have size (n,m+1) due to a bias term.
        """
        self.thetas = []
        for i, layer_length in enumerate(self.layer_lengths):
            #The last layer will have no theta
            if i == self.L:
                continue
            next_layer_length = self.layer_lengths[i+1]
            theta = np.zeros([next_layer_length,layer_length+1])
            for (x,y),val in np.ndenumerate(theta):
                theta[x,y] = self.get_rand()
            self.thetas.append(theta)

    def reset_ds(self):
        """
        ds will not contain any bias terms similar to zs
        """
        self.ds = [np.zeros([layer_length,1]) for layer_length in self.layer_lengths]

    def reset_deltas(self):
        self.deltas = []
        for i,layer_length in enumerate(self.layer_lengths):
            #The last layer will have no theta, so it will also have no delta
            if i == self.L:
                continue
            next_layer_length = self.layer_lengths[i+1]
            self.deltas.append(np.zeros([next_layer_length, layer_length+1]))

    def reset_Ds(self):
        self.Ds = []
        for i,layer_length in enumerate(self.layer_lengths):
            #The last layer will have no theta, so it will also have no D
            if i == self.L:
                continue
            next_layer_length = self.layer_lengths[i+1]
            self.deltas.append(np.zeros([next_layer_length, layer_length+1]))

    def prepend_one(self, col_vec):
        """
        Prepends a 1 to the top of any column vector
        effectively - ret = [1; col_vec]
        """
        return np.insert(col_vec,[0,0], 1,axis=0)

    def forward_prop(self,sample_x):
        """
        sample_x is an an individual
        sample_x = layer_lengths[0] x 1 ndarray
        """
        self.alphas[0] = self.prepend_one(sample_x)
        for i in range(0,len(self.layer_lengths)-1):
            layer_length = self.layer_lengths[i]
            self.zs[i+1] = np.dot(self.thetas[i],self.alphas[i])
            temp_alpha = expit(self.zs[i+1])
            #Dont add bias term for last element
            if i+1 == self.L:
                self.alphas[i+1] = temp_alpha
            else:
                self.alphas[i+1] = self.prepend_one(temp_alpha)

    def back_prop(self,sample_x, sample_y):
        """
        Will fail if forward_prop hasn't been performed
        """
        self.ds[self.L] = np.subtract(self.alphas[self.L],sample_y)
        for i in range(L-1,-1,-1):
            # d(l) = ((theta(l)' * d(l+1)) .* g'(a(l)))[2:end]
            self.ds[i] = np.multiply(np.dot(np.transpose(self.thetas[i]),self.ds[i+1]),self.d_expit_array(self.alphas[i]))[1:,:]
        for i in range(0,self.L-1):
            # delta(l) = delta(l) .+ d(l+1)*a(l)'
            self.deltas[i] = np.add(self.deltas[i], np.dot(self.ds[i+1], np.transpose(self.alphas[i])))

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
            self.reset_per_iteration()
            self.train_iteration(X,Y)

    def train_iteration(self,X,Y):
        m = X.shape[1]
        for i in range(0,m):
            self.reset_per_sample()
            self.train_sample(X[:,i], Y[:,i])
        for i in range(0,self.L):
            self.Ds[i] = np.add((1/m) * self.deltas[i], self.lmbda * self.thetas[i])

        #TODO some function that takes in theta and dJ/dTheta and gives the new theta
    
    def train_sample(self, sample_x, sample_y):
        """
        sample_x = self.layer_lengths[0] x 1 ndarray
        sample_y = self.layer_lengths[self.L] x1 ndarray
        """
        self.forward_prop(sample_x)
        self.back_prop(sample_x,sample_y)

    def error(self, expected, actual):
        """
        .5 * Sum(i) (expected[i] - actual[i])^2
        """
        diff = np.subtract(expected, actual)
        diff_squared = np.square(diff)
        return 0.5 * np.sum(diff_squared)
    
    def d_expit_single(self, input):
        """
        derivative calculation of sigmoid function for single input
        """
        return expit(input) * (1 - expit(input))
    
    def d_expit_array(self,input):
        """
        elementy by element derivative calculation of sigmoid function 
        """
        return np.multiply(expit(input), (1-expit(input)))




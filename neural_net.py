import numpy as np
import random
from scipy.special import expit

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
        self.init_alphas()
        self.init_zs()
        self.init_thetas()
        self.init_ds()
        self.init_deltas()
        self.init_Ds()
    
    def init_alphas(self):
        """
        Use an array of numpy arrays to allow us to vary layer_lengths.
        First value will always be a bias term except for in the last layer
        """
        self.alphas = []
        for (layer_length, i) in self.layer_lengths:
            #The last column will have no bias term
            if i == len(self.layer_lengths)-1:
                self.alphas.append(np.zeros([layer_length,1]))
            else:
                self.alphas.append(np.zeros([layer_length+1,1]))

    def init_zs(self):
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

    def init_thetas(self):
        """
        Theta[0] will map from a[0] to z[1]. Therefore, if layer 0 has size m, and layer 1 has size n,
        Then Theta[0] will have size (n,m+1) due to a bias term.
        """
        self.thetas = []
        for (layer_length, i) in self.layer_lengths:
            #The last layer will have no theta
            if i == len(self.layer_lengths)-1:
                continue
            next_layer_length = self.layer_lengths[i+1]
            theta = np.zeros([next_layer_length,layer_length+1])
            for (x,y) in numpy.ndenumerate(theta):
                theta[x,y] = self.get_rand()
            self.thetas.append(theta)

    def init_ds(self):
        """
        ds will not contain any bias terms similar to zs
        """
        self.ds = [np.zeros([layer_length,1]) for layer_length in self.layer_lengths]

    def init_deltas(self):
        self.deltas = []
        for (layer_length, i) in self.layer_lengths:
            #The last layer will have no theta, so it will also have no delta
            if i == self.num_layers-1:
                continue
            next_layer_length = self.layer_lengths[i+1]
            self.deltas.append(np.zeros([next_layer_length, layer_length+1]))

    def init_Ds(self):
        self.Ds = []
        for (layer_length, i) in self.layer_lengths:
            #The last layer will have no theta, so it will also have no D
            if i == self.num_layers-1:
                continue
            next_layer_length = self.layer_lengths[i+1]
            self.deltas.append(np.zeros([next_layer_length, layer_length+1]))

    def prepend_one(self, col_vec):
        """
        Prepends a 1 to the top of any column vector
        effectively - ret = [1; col_vec]
        """
        return np.insert(col_vec,[0,0], 1,axis=0)

    def forward_prop(self,sample):
        """
        sample = layer_lengths[0] x 1 ndarray
        """
        self.alphas[0] = self.prepend_one(sample)
        for i in range(0,len(self.layer_lengths)-1):
            layer_length = self.layer_lengths[i]
            self.zs[i+1] = np.dot(self.thetas[i],self.alphas[i])
            temp_alpha = expit(self.zs[i+1])
            #Dont add bias term for last element
            if i+1 == self.num_layers-1:
                self.alphas[i+1] = temp_alpha
            else:
                self.alphas[i+1] = self.prepend_one(temp_alpha)

    def train(self,X,num_iterations):
        #TODO
    
    def train_sample(self, sample):
        """
        sample = self.layer_lengths[0] x 1 ndarray
        """
        #TODO

    
    def d_expit_single(self, input):
        """
        derivative calculation of sigmoid function for single input
        """
        return expit(input) * (1 - expit(input))
    
    def d_expit_array(self,input):
        """
        elementy by element derivative calculation of sigmoid function gradient
        """
        return np.multiply(expit(input), (1-expit(input)))




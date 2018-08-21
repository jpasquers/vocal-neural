import numpy as np
import random

class Weights():
    def __init__(self, layer_lengths, epsilon):
        self.EPSILON = epsilon
        self.layer_lengths = layer_lengths
        self.num_layers = len(self.layer_lengths)
        self.L = self.num_layers - 1
        self.init_thetas()

    def init_thetas(self):
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

    def get_layer(self,layer):
        return self.thetas[layer]

    def update_layer(self,layer, theta):
        self.thetas[layer] = theta

    def get_rand(self):
        """
        Calculate a random value off of the given EPSILON value in the range [-self.EPSILON, self.EPSILON]
        """
        return random.random() * 2 * self.EPSILON - self.EPSILON
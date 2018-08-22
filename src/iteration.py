from src.sample import Sample
import numpy as np

class Iteration():
    def __init__(self, weights, X, Y, lmbda):
        self.weights = weights
        self.m = X.shape[1]
        self.lmbda = lmbda
        self.X = X
        self.Y = Y
        self.layer_lengths = self.weights.layer_lengths
        self.num_layers = len(self.layer_lengths)
        self.L = self.num_layers - 1
    
    def reset(self):
        self.reset_iteration_deltas()
        self.reset_partials()

    def calc_error_partials(self):
        ###
        self.reset()
        for i in range(0,self.m):
            sample = Sample(self.weights, self.X[:,[i]], self.Y[:,[i]])
            sample_deltas = sample.calc_sample_deltas()
            for i in range(0,self.L):
                self.iteration_deltas[i] = self.iteration_deltas[i] + sample_deltas[i]

        for i in range(0,self.L):
            # dJ/dTheta(i,j) = (1/m) * DELTA(i,j) + lambda * theta(i,j)
            self.partials[i] = np.add((1/self.m) * self.iteration_deltas[i], self.lmbda * self.weights.get_layer(i))

        return self.partials

    def reset_partials(self):
        self.partials = []
        for i,layer_length in enumerate(self.layer_lengths):
            #The last layer will have no theta, so it will also have no partial
            if i == self.L:
                continue
            next_layer_length = self.layer_lengths[i+1]
            self.partials.append(np.zeros([next_layer_length, layer_length+1]))

    def reset_iteration_deltas(self):
        self.iteration_deltas = []
        for i,layer_length in enumerate(self.layer_lengths):
            #The last layer will have no theta, so it will also have no delta
            if i == self.L:
                continue
            next_layer_length = self.layer_lengths[i+1]
            self.iteration_deltas.append(np.zeros([next_layer_length, layer_length+1]))

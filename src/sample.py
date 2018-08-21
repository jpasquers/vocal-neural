import numpy as np
from scipy.special import expit

class Sample():
    def __init__(self, weights, sample_x, sample_y):
        self.weights = weights
        self.layer_lengths = self.weights.layer_lengths
        self.num_layers = len(self.layer_lengths)
        #index of the last layer
        self.L = self.num_layers - 1
        self.sample_x = sample_x
        self.sample_y = sample_y

    def reset_matrices(self):
        self.reset_alphas()
        self.reset_zs()
        self.reset_ds()
        self.reset_deltas()

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

    def prepend_one(self, col_vec):
        """
        Prepends a 1 to the top of any column vector
        effectively - ret = [1; col_vec]
        """
        return np.insert(col_vec,0, np.array([[1]]),axis=0)

    def activation_fn(self,val):
        """
        This is brought into its own method to allow for overriding
        """
        return expit(val)

    def forward_prop(self):
        """
        sample_x is an an individual
        sample_x = layer_lengths[0] x 1 ndarray
        """
        self.alphas[0] = self.prepend_one(self.sample_x)
        for i in range(0,len(self.layer_lengths)-1):
            self.zs[i+1] = np.dot(self.weights.get_layer(i),self.alphas[i])
            temp_alpha = self.activation_fn(self.zs[i+1])
            #Dont add bias term for last element
            if i+1 == self.L:
                self.alphas[i+1] = temp_alpha
            else:
                self.alphas[i+1] = self.prepend_one(temp_alpha)

    def back_prop(self):
        """
        Will fail if forward_prop hasn't been performed
        """
        self.ds[self.L] = np.subtract(self.alphas[self.L],self.sample_y)
        for i in range(self.L-1,-1,-1):
            # d(l) = ((theta(l)' * d(l+1)) .* g'(a(l)))[2:end]
            g_prime = self.d_expit_array(self.alphas[i])
            dE_da = np.dot(np.transpose(self.weights.get_layer(i)),self.ds[i+1])
            self.ds[i] = np.multiply(dE_da,g_prime)[1:,:]
        
        for i in range(0,self.L):
            # delta(l) = d(l+1)*a(l)'
            self.deltas[i] = np.dot(self.ds[i+1], np.transpose(self.alphas[i]))

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


    def calc_sample_deltas(self):
        self.reset_matrices()
        self.forward_prop()
        self.back_prop()
        return self.deltas

    def get_outcome(self):
        """
        performs forward propogation and returns the value
        """
        self.reset_matrices()
        self.forward_prop()
        return self.alphas[self.L]
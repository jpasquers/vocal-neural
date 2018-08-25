from activation import Activation
from scipy.special import expit

class SigmoidActivation(Activation):
    def fn_single(self,input):
        return expit(input)
    
    def fn_array(self,input):
        return expit(input)

    def d_fn_single(self, input):
        
        """
        g'(x) = g(x) * (1 - g(x))
        derivative calculation of sigmoid function for single input
        """
        return expit(input) * (1 - expit(input))
    
    def d_fn_array(self,input):
        """
        g'(x) = g(x) * (1 - g(x))
        elementy by element derivative calculation of sigmoid function 
        """
        return np.multiply(expit(input), (1-expit(input)))

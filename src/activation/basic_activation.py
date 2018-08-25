from activation import Activation

class BasicActivation(Activation):

    def fn_single(self,input):
        return input
    
    def fn_array(self,input):
        return input
    
    def d_fn_single(self,input):
        return 1
    
    def d_fn_array(self,input):
        return np.ones(input.shape)
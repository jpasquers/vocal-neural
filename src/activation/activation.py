class Activation:
    """
    Used to apply different activation functions during forward/back propogation.
    In terms of the equations. This is the function transforming z -> a in a layer.
    a = g(z)
    """
    @abstractmethod
    def fn_single(self,input): raise NotImplementedError

    @abstractmethod
    def fn_array(self,input): raise NotImplementedError

    @abstractmethod
    def d_fn_single(self,input): raise NotImplementedError

    @abstractmethod
    def d_fn_array(self,input): raise NotImplementedError
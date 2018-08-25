class ErrorCalculator:
    @abstractmethod
    def last_layer_error(self,y,a): raise NotImplementedError

    @abstractmethod
    def d_last_layer_error(self,y,a): raise NotImplementedError


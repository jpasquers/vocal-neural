import numpy as np


class SquaredErrorCalculator(ErrorCalculator):

    def d_last_layer_error(self,y,a):
        return np.subtract(y,a)
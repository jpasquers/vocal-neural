import unittest
from src.neural_net import NeuralNet
from src.weights import Weights
from src.sample import Sample
import numpy as np

class BasicSample(Sample):
    def activation_fn(self,val):
        return val
    
    def d_activation_fn_single(self,val):
        return 1
    
    def d_activation_fn_array(self,val):
        return np.ones(val.shape)

class TestSample(unittest.TestCase):

    def setUp(self):
        self.layer_lengths = [3,3,3]
        self.weights = Weights(self.layer_lengths, 0.1)
        layer_0 = np.array([
            [1,2,1,2], #2
            [0,1,-2,1], #0
            [2,2,0.5,-1] #2
        ])
        layer_1 = np.array([
            [1,2,0,0], #5
            [0,2,0,1], #6
            [0,2,1,0] #4
        ])
        self.weights.update_layer(0,layer_0)
        self.weights.update_layer(1,layer_1)
        
        self.sample_x = np.array([[0.1],[0.2],[0.3]])
        self.sample_y = np.array([[2],[3],[2]])

    def test_get_outcome(self):
        sample = BasicSample(self.weights, self.sample_x, self.sample_y)
        result = sample.get_outcome()
        self.assertAlmostEqual(5,result[0][0])
        self.assertAlmostEqual(6,result[1][0])
        self.assertAlmostEqual(4,result[2][0])

    def test_calc_deltas(self):
        #ds[L] = [[-3][-3][-2]]
        #ds[1] = [[-16][-2][-3]]
        #ds[0] = [[-40][-13.5][-31]]
        #as[L] = [[5][6][4]]
        #as[1] = [[1][2][0][2]]
        #as[0] = [[1][0.1][0.2][0.3]]
        #deltas[1] = [[-3,-6,0,-6][-3,-6,0,-6][-2,-4,0,-4]]
        #deltas[0] = [[-16,-1.6,-3.2,-4.8][-2,-.2,-.4,-.6][-3,-.3,-.6,-.9]]

        sample = BasicSample(self.weights, self.sample_x, self.sample_y)
        deltas = sample.calc_sample_deltas()
        self.assertAlmostEqual(-3, deltas[1][0][0])
        self.assertAlmostEqual(-1.6, deltas[0][0][1])


if __name__ == '__main__':
    unittest.main()
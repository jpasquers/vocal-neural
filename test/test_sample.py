import unittest
from src.neural_net import NeuralNet
from src.weights import Weights
from src.sample import Sample
import numpy as np

class BasicSample(Sample):
    def activation_fn(self,val):
        return val

class TestSample(unittest.TestCase):

    def setUp(self):
        #
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
        self.sample_y = np.array([[0.2],[0.4],[0.6]])

    def test_get_outcome(self):
        sample = BasicSample(self.weights, self.sample_x, self.sample_y)
        result = sample.get_outcome()
        self.assertAlmostEqual(5,result[0][0])
        self.assertAlmostEqual(6,result[1][0])
        self.assertAlmostEqual(4,result[2][0])


if __name__ == '__main__':
    unittest.main()
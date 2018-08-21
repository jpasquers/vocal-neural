import unittest
from src.neural_net import NeuralNet
from weights import Weights
from sample import Sample
import numpy as np

class TestSample(unittest.TestCase):

    def setUp(self):
        #
        self.layer_lengths = [3,3,3]
        self.weights = Weights(self.layer_lengths, 0.1)
        layer_0 = np.ones([3,4])
        layer_1 = np.ones([3,4])
        self.weights.update_layer(0,layer_0)
        self.weights.update_layer(1,layer_1)
        
        self.sample_x = np.array([[0.1],[0.2],[0.3]])
        self.sample_y = np.array([[0.2],[0.4],[0.6]])

    def test_get_outcome(self):
        sample = Sample(self.weights, self.sample_x, self.sample_y)

if __name__ == '__main__':
    unittest.main()
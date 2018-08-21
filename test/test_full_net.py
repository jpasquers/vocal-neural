import unittest
from src.neural_net import NeuralNet
import numpy as np

class TestFullNetwork(unittest.TestCase):
    def setUp(self):
        self.neuralNet = NeuralNet([3,3])
        self.trainingSet_X = np.array([
            [.1, .04],
            [.2, .05],
            [.3, .06]
        ])
        self.trainingSet_Y = np.array([
            [.2, .08],
            [.4, .10],
            [.6, 12]
        ])
        self.neuralNet.train(self.trainingSet_X, self.trainingSet_Y, 1000)
        
    def test_somethin(self):
        self.testSet = np.array([
            [.07],
            [.09],
            [.11]
        ])
        result = self.neuralNet.get_outcome(self.testSet)
        print(self.neuralNet.thetas)
        print(result)
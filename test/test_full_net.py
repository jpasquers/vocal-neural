import unittest
from src.neural_net import NeuralNet
import numpy as np

class TestFullNetwork(unittest.TestCase):
    def setUp(self):
        self.neuralNet = NeuralNet([3,3,3])
        self.trainingSet_X = np.array([
            [1, 4, 7, 10],
            [2, 5, 8, 11],
            [3, 6, 9, 12]
        ])
        self.trainingSet_Y = np.array([
            [2, 8, 14, 20],
            [4, 10, 16, 22],
            [6, 12, 18, 24]
        ])
        self.neuralNet.train(self.trainingSet_X, self.trainingSet_Y, 400)
        
    def test_somethin(self):
        self.testSet = np.array([
            [3.5],
            [4.5],
            [5.5]
        ])
        result = self.neuralNet.get_outcome(self.testSet)
        print(self.neuralNet.thetas)
        print(result)
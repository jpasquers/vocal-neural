import unittest
from src.neural_net import NeuralNet
import numpy.ndenumerate as ndenumerate

class TestNeuralNet(unittest.TestCase):

    def setUp(self):
        self.neuralNet = NeuralNet([10,14,8,9])
    
    def test_reset_zs(self):
        self.assertEqual(4,len(self.neuralNet.zs))
        self.assertEqual(10, self.neuralNet.zs[0].shape[0])
        self.assertEqual(1, self.neuralNet.zs[0].shape[1])
        self.assertEqual(8, self.neuralNet.zs[2].shape[0])
    
    def test_reset_thetas(self):
        self.assertEqual(3, len(self.neuralNet.thetas))
        self.assertEqual(11, self.neuralNet.thetas[0].shape[1])
        self.assertEqual(14, self.neuralNet.thetas[0].shape[0])
        for (x,y),val in ndenumerate(self.neuralNet.thetas[0]):
            self.assertTrue(abs(val) < self.neuralNet.EPSILON)
if __name__ == '__main__':
    unittest.main()
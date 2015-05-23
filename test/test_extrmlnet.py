import sys, os
sys.path.insert(0, os.path.dirname('..'))
from extrmlnet import ExtrmlNet
import numpy as np
import pytest

class TestExtrmlNet:

    def setup_method(self, method):
        self.net = ExtrmlNet(2)
        self.net._input_weights = np.array([[2.0, 3.0], [1.0, 4.0]]) # two hidden neurons
        self.net._biases = np.array([5.0, 6.0]) # two biases
        self._input_data = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])


    def test_compute_hidden_outpus(self):
        h = self.net._compute_hidden_outputs(self._input_data) # 3 observations in input
        assert h.shape[0] == 3 and h.shape[1] == 2
        # testing that the hidden layer output is equal to g(w_j * x_i + b_j)
        np.testing.assert_almost_equal( self.net._activation_function(np.dot(np.array([2.0,3.0]),
                                                                             np.array([1.0,2.0])) + 5.0),
                                        h[0,0],
                                        decimal=5 )
        np.testing.assert_almost_equal( self.net._activation_function(np.dot(np.array([1.0,4.0]),
                                                                             np.array([3.0,4.0])) + 6.0),
                                        h[2,1],
                                        decimal=5 )

    def test_compute_hidden_weights_unregularized(self):
        self.net._compute_hidden_weights_unregularized(np.array([[1.0,2.0], [3.0, 4.0]]), np.array([1.0, -1.0]))
        np.testing.assert_almost_equal(self.net._output_weights, np.array([-3.0, 2.0]), decimal=5)

        net = ExtrmlNet(2, gamma=1.0)
        net._compute_hidden_weights_regularized(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, -1.0]))
        np.testing.assert_almost_equal(net._output_weights, np.array([0.5, -0.5]),decimal=5)

    def test_errors(self):
        with pytest.raises(ValueError) as error:
            net = ExtrmlNet(4)
            net.fit(np.ones((10, 3)), np.ones((10, 1)))
            net.predict(np.ones((11, 2)))
        assert error.value.message == 'X must have the same number of columns of the training matrix.'

        with pytest.raises(ValueError) as error:
            net = ExtrmlNet(4)
            net.fit(np.ones((10, 3)), np.ones((11,1)))
            net.predict(np.ones((10, 2)))
        assert error.value.message == 'X and T must have the same number of rows.'

    def test_regularization(self):
        net = ExtrmlNet(4,gamma=1)
        assert net._compute_hidden_weights == net._compute_hidden_weights_regularized
        net = ExtrmlNet(4)
        assert net._compute_hidden_weights == net._compute_hidden_weights_unregularized

    def test_predict(self):
        net = ExtrmlNet(1, random_seed=2)
        X = np.ones((10, 10))
        y = np.ones((10, 2))
        net.fit(X,y)
        np.testing.assert_array_almost_equal(net.predict(X), y)
        net = ExtrmlNet(1,random_seed=1)
        X = np.ones((10, 10))
        y = np.zeros((10, 2))
        net.fit(X,y)
        np.testing.assert_array_equal(net.predict(X) ,y)
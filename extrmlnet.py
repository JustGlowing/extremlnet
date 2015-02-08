from numpy.random import rand, seed
from numpy.linalg import pinv
from numpy import dot, zeros, exp, eye


class ExtrmlNet:
    """
     Single Layer Feedforward Neural Network trained with the Extreme Learning Machine algorithm.

     This software have been inspired by the following papers:

     1) Guang-bin Huang and Qin-yu Zhu and Chee-kheong Siew.
        Extreme learning machine: A new learning scheme of feedforward neural networks.
        In Proceedings of International Joint Conference on Neural Networks. 2006.

     2) Wenbin Zheng, Yuntao Qian, Huijuan Lu.
        Text categorization based on regularization extreme learning machine.
        In Neural Computing and Applications. 2013.
    """

    def __init__(self, num_hidden_neurons=10, gamma=None, random_seed=None):
        """
         Single layer feedforward neural network.
         Input:
          - num_hidden_neurons, number of neurons in the hidden layer.
          - gamma, regularization parameter. This parameter is able to reduce overfitting.
                   The greater this value will be, the less overfitting you have but also greater bias.
          - random_seed, random seed to use when generating the weights.
                         (Notice if the seed is passed the contructor will call
                         numpy.seed() affecting your future numpy.random calls)
        """
        if random_seed:
            seed(random_seed)
        if gamma:
            self._compute_hidden_weights = self._compute_hidden_weights_regularized
            self._gamma = gamma
        else:
            self._compute_hidden_weights = self._compute_hidden_weights_unregularized
        self._num_hidden_neurons = num_hidden_neurons

    def fit(self, X, T):
        """
         Train the neural network.
         Input:
          - X, numpy array or sparse matrix of shape [n_samples,n_inputs]. (Training Data)
          - T, numpy array or sparse matrix of shape [n_samples,n_outputs]. (Target values)
        n_inputs will be the number of input neurons of the network and n_outputs will be the number of output neurons,
        Each row of X represents an observation and each column a feature.
        While each row of T represent the target value(s).
        """
        self._num_features = 1
        if len(X.shape) > 1:
            self._num_features = X.shape[1]
        if len(X) != len(T):
            raise ValueError('X and T must have the same number of rows.')
        # the input weights and the biases are never modified after the initialization
        self._input_weights = rand(self._num_hidden_neurons, self._num_features) * 2 - 1
        self._biases = rand(self._num_hidden_neurons) * 2 - 1
        hidden_outputs = self._compute_hidden_outputs(X)
        self._compute_hidden_weights(hidden_outputs, T)

    def predict(self, X):
        """
         Compute the output of the neural network.
         Input:
          - X, numpy array or sparse matrix of shape [n_samples,number_of_features]. (Samples)
         Output:
          - P array, shape = (n_samples,n_outputs) (Predicted values)
        """
        if (len(X.shape) > 1 and self._num_features != X.shape[1]) or (len(X.shape) == 1 and self._num_features != 1):
            raise ValueError('X must have the same number of columns of the training matrix.')
        hidden_outputs = self._compute_hidden_outputs(X)
        return dot(hidden_outputs, self._output_weights)

    def _compute_hidden_weights_unregularized(self, hidden_outputs, T):
        self._output_weights = dot(pinv(hidden_outputs), T)

    def _compute_hidden_weights_regularized(self, hidden_outputs, T):
        _inv = pinv(dot(hidden_outputs.T, hidden_outputs)+eye(self._num_hidden_neurons)*self._gamma)
        self._output_weights = dot(dot(_inv, hidden_outputs.T), T) # see reference 2 for this implementation

    def _compute_hidden_outputs(self, X):
        """
         Return the hidden output matrix H (num_observation by num_hidden_neurons) where
            H(i,j) = g(w_j * x_i + b_j)
         where x_j and b_j are the weights and the bias of the j-th hidden neuron, x_i is the i-th observation and b_j
         and g is the activation function.
        """
        hidden_outputs = zeros((X.shape[0], self._num_hidden_neurons))
        for i, x in enumerate(X):
            for j, w in enumerate(self._input_weights):
                hidden_outputs[i,j] = self._activation_function(dot(w,x) + self._biases[j])
        return hidden_outputs

    def _activation_function(self, x):
        """ Hidden neurons sigmoidal activation function. """
        return 1 / (1 + exp(-x))
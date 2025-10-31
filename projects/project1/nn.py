'''
Neural network implementation module.
'''
import numpy as np


class Layer:
    '''
    Represents a single layer of a neural network, along with its transfer function.
    '''
    def __init__(self, weight_matrix, bias, transfer_func=lambda x: np.maximum(0, x)):
        '''
        Construct a layer with M inputs and N outputs.

        Args:
            weight_matrix: numpy array of shape=(N, M)
            bias: numpy array of shape=(N, )
            transfer_func: the transfer function
        '''
        self.weight_matrix = weight_matrix
        self.bias = bias
        self.transfer_func = transfer_func


    def evaluate(self, input_vec):
        '''
        Evaluates the layer on a single input vector.

        Args:
            input_vec: numpy array of shape=(M, )

        Returns:
            the output value of the layer for the given input, of shape=(N, )
        '''
        return self.transfer_func(np.matmul(input_vec, self.weight_matrix.T) + self.bias)


class NN:
    '''
    Represents a neural network.
    '''
    def __init__(self, shape, weight_vector=None):
        '''
        Constructs a neural network.

        Args:
            shape: array representing the number of the inputs of each layer.
            weight_vector: array containing all weights of the neural network.
            If none, it will be sampled from a N(0, 0.01) distribution.
        '''
        rng = np.random.default_rng()

        if weight_vector is None:
            weight_cnt = sum(shape[i - 1] * shape[i] + shape[i] for i in range(1, len(shape)))
            weight_vector = rng.normal(0.0, 0.01, weight_cnt)
        
        self.shape = shape
        
        self.set_weight_vector(weight_vector)


    def set_weight_vector(self, weight_vector):
        '''
        Sets the weights of the neural network.

        Args:
            weight_vector: array of weights
        '''
        self.layers = []

        self.weight_vector = weight_vector

        idx = 0
        for i in range(1, len(self.shape)):
            in_cnt = self.shape[i - 1]
            out_cnt = self.shape[i]

            w_mat = np.reshape(weight_vector[idx : idx + in_cnt * out_cnt], (out_cnt, in_cnt))
            idx += in_cnt * out_cnt
            bias_vec = weight_vector[idx : idx + out_cnt]
            idx += out_cnt

            if i != len(self.shape) - 1:
                self.layers.append(Layer(w_mat, bias_vec))
            else:
                self.layers.append(Layer(
                    w_mat, bias_vec,
                    lambda x: 1 / (1 + np.exp(-x))
                ))


    def get_weight_vector(self):
        '''
        Get the weights of the neural network.

        Returns:
            array of all weights.
        '''
        return self.weight_vector


    def evaluate(self, X):
        '''
        Evaluates the neural network for the given dataset.

        Args:
            X: input dataset
        
        Returns:
            output of neural network
        '''
        for layer in self.layers:
            #print(f"evaluating f({X})")
            y = layer.evaluate(X)
            X = y
        
        return y


    def error(self, x_train, y_train):
        '''
        Calculate the error (binary cross entropy) for the given dataset.

        Args:
            x_train: training dataset
            y_train: training outputs (binary label probabilities)

        Returns:
            mean of binary cross entropy for each datapoint.
        '''
        y_pred = self.evaluate(x_train)
        
        return -np.mean(
            y_train * np.maximum(np.log2(y_pred), -100) + 
            (1 - y_train) * np.maximum(np.log2(1 - y_pred), -100)
        )
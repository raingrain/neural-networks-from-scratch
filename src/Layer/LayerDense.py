import numpy as np


# Dense layer
class LayerDense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0,
                 weight_regularizer_l2=0, bias_regularizer_l1=0,
                 bias_regularizer_l2=0):
        self.d_inputs = None
        self.d_biases = None
        self.d_weights = None
        self.output = None
        self.inputs = None
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, d_values):
        # Gradients on parameters
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)
        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            d_l1 = np.ones_like(self.weights)
            d_l1[self.weights < 0] = -1
            self.d_weights += self.weight_regularizer_l1 * d_l1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.d_weights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            d_l1 = np.ones_like(self.biases)
            d_l1[self.biases < 0] = -1
            self.d_biases += self.bias_regularizer_l1 * d_l1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.d_biases += 2 * self.bias_regularizer_l2 * self.biases
        # Gradient on values
        self.d_inputs = np.dot(d_values, self.weights.T)

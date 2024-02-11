import numpy as np


# ReLU activation
class ActivationReLU:

    def __init__(self):
        self.output = None
        self.inputs = None

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.d_inputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.d_inputs[self.inputs <= 0] = 0

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs

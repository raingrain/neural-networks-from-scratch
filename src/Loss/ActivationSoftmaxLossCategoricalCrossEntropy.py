import numpy as np


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class ActivationSoftmaxLossCategoricalCrossEntropy:

    def __init__(self):
        self.d_inputs = None

    # Backward pass
    def backward(self, d_values, y_true):
        # Number of samples
        samples = len(d_values)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.d_inputs = d_values.copy()
        # Calculate gradient
        self.d_inputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples

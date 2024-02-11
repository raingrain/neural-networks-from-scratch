import numpy as np

from src.Loss.Loss import Loss


# Mean Squared Error loss
class LossMeanSquaredError(Loss):  # L2 loss

    # Forward pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, d_values, y_true):
        # Number of samples
        samples = len(d_values)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(d_values[0])
        # Gradient on values
        self.d_inputs = -2 * (y_true - d_values) / outputs
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples

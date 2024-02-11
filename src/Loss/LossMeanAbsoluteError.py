import numpy as np

from src.Loss.Loss import Loss


# Mean Absolute Error loss
class LossMeanAbsoluteError(Loss):  # L1 loss

    def __init__(self):
        super().__init__()
        self.d_inputs = None

    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, d_values, y_true):
        # Number of samples
        samples = len(d_values)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(d_values[0])
        # Calculate gradient
        self.d_inputs = np.sign(y_true - d_values) / outputs
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples

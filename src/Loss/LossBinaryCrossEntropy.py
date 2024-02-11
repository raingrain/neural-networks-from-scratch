import numpy as np

from src.Loss.Loss import Loss


# Binary cross-entropy loss
class LossBinaryCrossEntropy(Loss):

    def __init__(self):
        super().__init__()
        self.d_inputs = None

    # Forward pass
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Calculate sample-wise loss
        sample_losses = -(
                y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(
            1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_d_values = np.clip(dvalues, 1e-7, 1 - 1e-7)
        # Calculate gradient
        self.d_inputs = -(y_true / clipped_d_values - (1 - y_true) / (
                1 - clipped_d_values)) / outputs
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples

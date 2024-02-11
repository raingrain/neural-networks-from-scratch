# Linear activation
class ActivationLinear:

    def __init__(self):
        self.d_inputs = None
        self.output = None
        self.inputs = None

    # Forward pass
    def forward(self, inputs, training):
        # Just remember values
        self.inputs = inputs
        self.output = inputs

    # Backward pass
    def backward(self, d_values):
        # derivative is 1, 1 * d_values = d_values - the chain rule
        self.d_inputs = d_values.copy()

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs

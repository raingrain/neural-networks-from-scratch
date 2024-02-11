# Input "layer"
class LayerInput:

    def __init__(self):
        self.output = None

    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs

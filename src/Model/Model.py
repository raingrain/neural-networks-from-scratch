from src.Activation.ActivationSoftmax import ActivationSoftmax
from src.Layer.LayerInput import LayerInput
from src.Loss.ActivationSoftmaxLossCategoricalCrossEntropy import \
    ActivationSoftmaxLossCategoricalCrossEntropy
from src.Loss.LossCategoricalCrossEntropy import LossCategoricalCrossEntropy


# Model class
class Model:

    def __init__(self):
        self.output_layer_activation = None
        self.trainable_layers = None
        self.input_layer = None
        self.accuracy = None
        self.optimizer = None
        self.loss = None
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    # Set loss, optimizer and accuracy
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # Finalize the model
    def finalize(self):
        # Create and set the input layer
        self.input_layer = LayerInput()
        # Count all the objects
        layer_count = len(self.layers)
        # Initialize a list containing trainable layers:
        self.trainable_layers = []
        # Iterate the objects
        for i in range(layer_count):
            # If it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            # The last layer - the next object is the loss
            # Also let's save aside the reference to the last object
            # whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)
        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        if isinstance(self.layers[-1], ActivationSoftmax) and isinstance(
                self.loss, LossCategoricalCrossEntropy):
            # Create an object of combined activation
            # and loss functions
            self.softmax_classifier_output = ActivationSoftmaxLossCategoricalCrossEntropy()

    # Train the model
    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        # Initialize accuracy object
        self.accuracy.init(y)
        # Main training loop
        for epoch in range(1, epochs + 1):
            # Perform the forward pass
            output = self.forward(X, training=True)
            # Calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y,
                                                                 include_regularization=True)
            loss = data_loss + regularization_loss
            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)
            # Perform backward pass
            self.backward(output, y)
            # Optimize (update parameters)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()
            # Print a summary
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')
        # If there is the validation data
        if validation_data is not None:
            # For better readability
            X_val, y_val = validation_data
            # Perform the forward pass
            output = self.forward(X_val, training=False)
            # Calculate the loss
            loss = self.loss.calculate(output, y_val)
            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(
                output)
            accuracy = self.accuracy.calculate(predictions, y_val)
            # Print a summary
            print(f'validation, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}')

    # Performs forward pass
    def forward(self, X, training):
        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)
        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        # "layer" is now the last object from the list,
        # return its output
        return layer.output

    # Performs backward pass
    def backward(self, output, y):
        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set d_inputs property
            self.softmax_classifier_output.backward(output, y)
            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set d_inputs in this object
            self.layers[-1].d_inputs = self.softmax_classifier_output.d_inputs
            # Call backward method going through
            # all the objects but last
            # in reversed order passing d_inputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.d_inputs)
            return
        # First call backward method on the loss
        # this will set d_inputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)
        # Call backward method going through all the objects
        # in reversed order passing d_inputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.d_inputs)

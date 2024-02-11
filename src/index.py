import nnfs

from nnfs.datasets import spiral_data

from src.Accuracy.AccuracyCategorical import AccuracyCategorical
from src.Activation.ActivationReLU import ActivationReLU
from src.Activation.ActivationSoftmax import ActivationSoftmax
from src.Layer.LayerDense import LayerDense
from src.Layer.LayerDropout import LayerDropout
from src.Loss.LossCategoricalCrossEntropy import LossCategoricalCrossEntropy
from src.Model.Model import Model
from src.Optimizer.OptimizerAdam import OptimizerAdam

# Fixed random seeds make data consistent
nnfs.init()

# Create dataset
X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# Instantiate the model
model = Model()

# Add layers
model.add(
    LayerDense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(ActivationReLU())
model.add(LayerDropout(0.1))
model.add(LayerDense(512, 3))
model.add(ActivationSoftmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=LossCategoricalCrossEntropy(),
    optimizer=OptimizerAdam(learning_rate=0.05, decay=5e-5),
    accuracy=AccuracyCategorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=10000,
            print_every=100)

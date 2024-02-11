import numpy as np


# Softmax activation
class Activation_Softmax:

    def __init__(self):
        self.output = None

    # Forward pass
    def forward(self, inputs):
        # 对于每一行的一组数据，先减去最大值，然后计算e的次方
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # 对每个样本进行归一化处理，总和标准化
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        # 输出
        self.output = probabilities

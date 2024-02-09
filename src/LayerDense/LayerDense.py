import numpy as np


# 全连接层 or 稠密层 or 致密层
class Layer_Dense:

    # 随机初始化
    def __init__(self, n_inputs, n_neurons):
        # 随机初始化偏差和权重
        # 初始化每一个神经元对各个输入的权重
        # 我们将把高斯分布的权重乘以0.01，以生成数值小数数量级的值
        # 否则在训练过程中，由于起始值相对于训练中进行的更新过大，模型会花费更多时间来适应数据
        # 这里的想法是以足够小的非零值启动模型，以确保它们不会对训练产生影响
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # 初始化每一个神经元的偏差
        self.biases = np.zeros((1, n_neurons))
        self.output = None

    # 前向传播
    def forward(self, inputs):
        # 根据输入、权重和偏差计算输出值
        self.output = np.dot(inputs, self.weights) + self.biases
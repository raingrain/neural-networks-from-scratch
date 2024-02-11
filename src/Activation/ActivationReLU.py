import numpy as np


# ReLU 激活函数
# 一般来说用于线性
# y = 0, x <= 0
# y = x, x > 0
class Activation_ReLU:

    def __init__(self):
        self.output = None

    # 前向传播
    def forward(self, inputs):
        # 根据输入计算输出值
        self.output = np.maximum(0, inputs)

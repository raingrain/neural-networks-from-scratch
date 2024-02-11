import numpy as np


# 普通损失类
class Loss:

    # 计算数据和正则化损失
    # 给定的模型输出和真实值
    def calculate(self, output, y):
        # 计算样本损失
        sample_losses = self.forward(output, y)

        # 计算平均损失
        data_loss = np.mean(sample_losses)

        # 返回损失
        return data_loss

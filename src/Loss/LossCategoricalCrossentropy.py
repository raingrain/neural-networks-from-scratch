import numpy as np

from src.Loss.Loss import Loss


# 交叉熵损失
class Loss_CategoricalCrossentropy(Loss):

    # 前向传播
    def forward(self, y_pred, y_true):
        # 一个batch中有几个sample
        samples = len(y_pred)

        # 剪裁数据以防止被0除
        # 剪裁两侧以不将均值拖向任何值
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 目标值的概率 - 仅当类别标签
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # 掩码值 - 仅适用于一个热编码标签
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # 损失值
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

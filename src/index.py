import nnfs
from nnfs.datasets import spiral_data

from src.LayerDense.LayerDense import Layer_Dense

# 锁定随机种子，使得数据固定便于调试
nnfs.init()

# 创建数据集
X, y = spiral_data(samples=100, classes=3)

# 使用2个输入特征和3个输出值创建全连接层
dense1 = Layer_Dense(2, 3)

# 通过该层向前传递我们的训练数据
dense1.forward(X)

# 让我们看看前几个示例的输出：
print(dense1.output[:5])
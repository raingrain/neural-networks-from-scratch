from nnfs import init
from nnfs.datasets import spiral_data

from src.Activation.ActivationReLU import Activation_ReLU
from src.Activation.ActivationSoftmax import Activation_Softmax
from src.Layer.LayerDense import Layer_Dense

# 锁定随机种子，使得数据固定便于调试
init()

# 创建数据集
X, y = spiral_data(samples=100, classes=3)

# 使用2个输入特征和3个输出值创建全连接层
dense1 = Layer_Dense(2, 3)

# 创建ReLU激活（与密集层一起使用）：
activation1 = Activation_ReLU()

# 创建具有3个输入特征的第二个密集层（如我们在此处获取上一层的输出）和3个输出值
dense2 = Layer_Dense(3, 3)

# 创建Softmax激活（用于密集层）：
activation2 = Activation_Softmax()

# 通过该层向前传递我们的训练数据
dense1.forward(X)

# 继续前向传播
# 这里取第一个稠密层的输出
activation1.forward(dense1.output)

# 继续前向传播
# 它以第一层激活函数的输出作为输入
dense2.forward(activation1.output)

# 继续前向传播
# 这里取第二个稠密层的输出
activation2.forward(dense2.output)

# 让我们看看前几个示例的输出：
print(activation2.output[:5])

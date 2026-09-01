# -*- coding: utf-8 -*-
"""
第一次 PyTorch 实操：让模型学会 y = 2x + 1

现在文件里只有"第一步：造数据"。
每一行都写了注释，请一行一行读懂，然后完成底部的小任务。

运行方式（在项目根目录 F:\Projects\bms judge 下）：
    .venv\Scripts\python.exe learn_pytorch\linear_demo.py
"""

import torch  # 这行是"导入 PyTorch"，之后才能用 torch 开头的功能

torch.manual_seed(0)  # 让"随机数"每次运行都一样（调试时很有用；去掉就恢复随机）


# ---------- 第一步：创建数据 ----------

N = 10  # x 从 1 数到 N。N 是一个普通整数（Python 变量）

# torch.arange(开始, 结束, dtype=...) 生成一串数：开始, 开始+1, ..., 结束-1
# 所以 arange(1, N+1) = 1, 2, 3, ..., 100
# dtype=torch.float32 表示"用浮点数存"（后面做乘法、训练都需要浮点，整数不行）
x = torch.arange(1, N + 1, dtype=torch.float32)

# y = 2x + 1：这里 PyTorch 张量可以直接做算术，结果是另一个张量
y = 3 * x + 2

# .shape 是张量的"形状"（每一维有多大）。这里是一维，长度 N
print("x shape:", x.shape)
print("y shape:", y.shape)

# 打印前 5 个，确认数据长什么样（x 是 1..5，y 应该是 3,5,7,9,11）
print("x[:5] =", x[:5])
print("y[:5] =", y[:5])


# ---------- 小任务（你来动手） ----------

# 任务：把上面的 N = 100 改成 N = 10，然后重新运行。
# 观察：shape 变了没有？x[:5] 和 y[:5] 变了吗？
# 再把下面这行注释打开（删掉 #），自己打印 y 的最后 3 个数：
print("y[-3:] =", y[-3:])


# 先别往下写。等这一步跑通、你回答了上面的问题，我们再进入第二步（模型）。


# ---------- 第二步：创建模型 ----------

import torch.nn as nn  # nn 里放着"神经网络零件"（Linear 层等）


# class = 定义一种"东西"；LinearModel 就是我们要的模型
# nn.Module 是 PyTorch 模型的"基类"，我们写的模型要继承它
class LinearModel(nn.Module):
    def __init__(self):
        # __init__ 是"创建这个模型时自动执行"的代码
        super().__init__()                # 先初始化基类（固定写法）
        self.linear = nn.Linear(1, 1)     # 一层线性层：输入 1 个数、输出 1 个数
        # nn.Linear 内部有两个"参数"：权重 w（weight）和偏置 b（bias），
        # 创建时是随机数。我们的目标就是通过训练把它们变成 w≈2、b≈1。

    def forward(self, x):
        # forward 决定"输入 x 时，模型怎么算出输出"（前向计算）
        return self.linear(x)  # 把 x 送进 linear 层，linear 会算 w*x + b


model = LinearModel()  # 创建模型（会执行 __init__）

# 看看训练前 w、b 是什么（应该是随机数，不是 2 和 1）
print("初始 w:", model.linear.weight.item())
print("初始 b:", model.linear.bias.item())

# 让模型预测 x=5 应该是什么。正确答案是 11，但没训练时是随机数
for i in range(0,2):
    print("模型预测", model(torch.tensor([float(i)])).item())


# ---------- 第三步：loss 和 optimizer ----------

# 注意：nn.Linear(1, 1) 期望输入形状是 (N, 1)——最后一维必须是 1
# 我们的 x 目前是 (N,)，所以要先"reshape"成 (N, 1)。-1 表示"这一维自动算"
x_col = x.reshape(-1, 1)
y_col = y.reshape(-1, 1)

# loss（损失）：衡量"预测 ŷ 和真实 y 差多少"。MSE = 平均 (ŷ - y)²
# TODO 你来写这一行：
loss_fn = nn.MSELoss(1)


# optimizer（优化器）：决定"根据梯度把 w、b 往哪挪、挪多少"
# model.parameters() 就是模型里所有可学习的参数（w 和 b）
# lr（学习率）= 每次挪动的步子大小
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 试算一次 loss：拿模型现在的（乱猜的）预测和正确答案比一比
y_pred = model(x_col)
print("当前 loss（还没训练，应该比较大）:", loss_fn(y_pred, y_col).item())


# ---------- 第四步：训练循环（你来写） ----------

# 循环体需要 5 行，顺序如下（把下面的 TODO 替换成真正的代码）：
for epoch in range(1000):
    y_pred = model(x_col)
    loss = loss_fn(y_pred, y_col)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch {epoch}: loss={loss.item():.4f}  "
              f"w={model.linear.weight.item():.4f}  b={model.linear.bias.item():.4f}")

# 跑完看看：loss 应该从 165 左右一路降到接近 0，w 接近 2，b 接近 1
print("训练后 w:", model.linear.weight.item(), " b:", model.linear.bias.item())

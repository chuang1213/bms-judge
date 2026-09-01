"""最简单的 baseline：MLP（回归）。

结构：输入(26) → Linear(64) → ReLU → Dropout → Linear(32) → ReLU → Linear(1)。
参数全部通过训练学习：两层的 weight/bias 与 dropout 无关（dropout 只在训练时
随机置零激活，测试时关闭）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 64, hidden2: int = 32,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 (batch, input_dim) → 输出 (batch, 1)
        return self.net(x)

from __future__ import annotations

import torch
import torch.nn as nn


class SequenceCNN(nn.Module):
    def __init__(self, seq_dim: int = 4, embed_dim: int = 32,
                 conv_channels: int = 64):
        """
        Args:
            seq_dim: 输入特征维度 (固定为4)
            embed_dim: 每个时间步的嵌入维度
            conv_channels: 卷积输出通道数，也是池化后的特征维度
        """
        super().__init__()
        self.embed = nn.Linear(seq_dim, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(conv_channels, 1)

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取卷积后的全局池化特征向量。
        Args:
            x: [B, L, 4]
        Returns:
            [B, conv_channels]
        """
        # 逐时间步升维
        x = self.embed(x)                 # [B, L, embed_dim]
        # 转置为 Conv1d 需要的格式
        x = x.transpose(1, 2)             # [B, embed_dim, L]
        # 两层卷积 + ReLU
        x = self.relu(self.conv1(x))      # [B, conv_channels, L]
        x = self.relu(self.conv2(x))      # [B, conv_channels, L]
        # 在长度维度上全局平均池化
        x = x.mean(dim=2)                 # [B, conv_channels]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，输出预测的 difficulty 标量。
        Args:
            x: [B, L, 4]
        Returns:
            [B, 1]
        """
        features = self.pool(x)           # [B, conv_channels]
        return self.fc(features)          # [B, 1]
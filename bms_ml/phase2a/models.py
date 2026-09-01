"""Phase 2A 最小模型：小 CNN（无 Transformer / 无 attention）。

GridEncoder：输入 [B, 2, T, 8] 网格 → 局部特征图 [B, C, T, 8] + 全局池化向量 [B, C]。
masked reconstruction：局部特征 + 全局特征广播拼接 → 1x1 conv → [B, 2, T, 8] logits。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .grid_data import N_CH, N_LANES, T


class GridEncoder(nn.Module):
    def __init__(self, channels: tuple = (32, 64, 64),
                 time_kernel: int = 7, lane_kernel: int = 3,
                 global_dim: int = 64):
        super().__init__()
        c0, c1, c2 = channels
        self.conv1 = nn.Conv2d(N_CH, c0, (time_kernel, lane_kernel),
                               padding=(time_kernel // 2, lane_kernel // 2))
        self.conv2 = nn.Conv2d(c0, c1, (time_kernel, lane_kernel),
                               padding=(time_kernel // 2, lane_kernel // 2))
        self.conv3 = nn.Conv2d(c1, c2, (time_kernel, lane_kernel),
                               padding=(time_kernel // 2, lane_kernel // 2))
        self.relu = nn.ReLU()
        self.global_fc = nn.Sequential(
            nn.Linear(c2, global_dim), nn.ReLU())
        # 局部特征 + 全局广播 → 每 cell 预测 onset/hold
        self.head = nn.Conv2d(c2 + global_dim, N_CH, 1)

    def features(self, x: torch.Tensor):
        """x: [B,2,T,8] -> (local [B,C,T,8], global [B,C])"""
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        g = h.mean(dim=(2, 3))                       # [B,C]
        return h, g

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        _, g = self.features(x)
        return g

    def cell_features(self, x: torch.Tensor) -> torch.Tensor:
        """局部特征 + 全局广播拼接 → [B, c2+global, T, L]。"""
        local, g = self.features(x)
        gb = self.global_fc(g)[:, :, None, None]
        return torch.cat([local, gb.expand(-1, -1, T, N_LANES)], dim=1)

    def forward_mask(self, x: torch.Tensor):
        """masked reconstruction 前向：返回 (logits [B,2,T,8], global [B,C])。"""
        fused = self.cell_features(x)
        return self.head(fused), self.pool(x)

    def forward(self, x: torch.Tensor):
        return self.forward_mask(x)[0]


class NextWindowModel(nn.Module):
    """T2 对照：从当前窗口的全局池化向量预测下一窗口网格（仅全局信息，刻意最小化）。"""

    def __init__(self, encoder: GridEncoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.global_fc[0].out_features, T * N_LANES * N_CH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.encoder.pool(x)
        return self.head(g).view(-1, N_CH, T, N_LANES)


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

"""PyTorch Dataset：把 manifest 记录包装成 (features, label, meta)。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class ChartDataset(Dataset):
    """每个样本：stats 特征向量 + 标量 label（可选）。

    输入 tensor 形状: (num_features,)，即 (26,)；
    输出 label 形状: 标量 float（单表难度等级，作为有序数值使用）。
    """

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        x = torch.from_numpy(rec["features"].astype(np.float32))
        y = rec.get("label")
        if y is None:
            y = torch.tensor(float("nan"))
        else:
            y = torch.tensor(float(y), dtype=torch.float32)
        meta = {
            "path": rec["path"],
            "title": rec.get("title", ""),
            "sha256": rec.get("sha256", ""),
        }
        return x, y, meta


def make_dataloader(records, batch_size, shuffle, num_workers=0):
    ds = ChartDataset(records)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

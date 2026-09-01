"""BMS 难度表（難易度表）标签接口。

难度表格式（beatoraja / GLAssist / BeMusicSeeker 通用）为
头部 JSON（name/symbol/data_url/level_order）+ 数据 JSON（ChartInfo 数组，
每项至少含 md5 或 sha256 之一与 level 字符串）。
本模块只负责下载/解析难度表、按 md5/sha256 查 level，不做跨表换算。
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DifficultyTable:
    name: str
    symbol: str
    source: str
    level_order: List[str] = field(default_factory=list)
    md5_to_level: Dict[str, str] = field(default_factory=dict)
    sha256_to_level: Dict[str, str] = field(default_factory=dict)
    entries: int = 0

    def lookup(self, md5: Optional[str], sha256: Optional[str]) -> Optional[str]:
        """返回该表对某谱面的 level 字符串；未收录返回 None。"""
        if sha256 and sha256 in self.sha256_to_level:
            return self.sha256_to_level[sha256]
        if md5 and md5 in self.md5_to_level:
            return self.md5_to_level[md5]
        return None

    def level_value(self, level: str) -> Optional[float]:
        """level 字符串转数值；'?' 等未知值返回 None（不参与训练）。"""
        if level in ("?", "0-", "☆?", "★?"):
            return None
        try:
            return float(level)
        except ValueError:
            return None


def parse_table(header_path: str, data_path: str, source: str = "") -> DifficultyTable:
    with open(header_path, "r", encoding="utf-8") as f:
        h = json.load(f)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    name = h.get("name", os.path.basename(header_path))
    table = DifficultyTable(
        name=name,
        symbol=h.get("symbol", ""),
        source=source or h.get("data_url", ""),
        level_order=[str(x) for x in h.get("level_order", [])],
    )
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    for item in data:
        if not isinstance(item, dict):
            continue
        level = item.get("level")
        if level is None:
            continue
        md5 = item.get("md5")
        sha256 = item.get("sha256")
        if md5:
            table.md5_to_level[str(md5).lower()] = str(level)
        if sha256:
            table.sha256_to_level[str(sha256).lower()] = str(level)
        table.entries += 1
    return table


def fetch_table_json(url: str, dest: str, proxy: Optional[str] = None) -> None:
    """下载难度表 JSON（可用 127.0.0.1:7897 代理）。"""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({
            "http": proxy,
            "https": proxy,
        }))
    opener = urllib.request.build_opener(*handlers)
    opener.addheaders = [("User-Agent", "bms-ml/0.1")]
    with opener.open(url, timeout=60) as r:
        raw = r.read()
    with open(dest, "wb") as f:
        f.write(raw)

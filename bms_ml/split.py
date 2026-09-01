"""同曲分组划分（防止同曲差分泄漏）。

策略：
1. group key = 归一化后的 (artist, title)；缺任一字段时退化为所在目录（package）。
2. 按 group 划分 train/val，同一曲目的所有差分永远在同一侧。
3. 输出分组统计与泄漏检查（跨 split 的同 title 数应为 0）。
"""

from __future__ import annotations

import re
import unicodedata
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower()
    s = re.sub(r"[\W_]+", "", s)
    return s


# 常见难度/版本后缀（剥离后得到"基础曲名"）
_DIFF_TOKENS = [
    "another", "hyper", "normal", "hard", "hardest", "easy", "beginner",
    "god", "hades", "insane", "maniaq", "extreme", "afother", "ex", "ln",
    "light", "pocket", "lastboss", "lumiere", "sa", "ecstacy", "presea",
    "main", "seiryu", "pham", "9dan", "plus", "myth", "hina", "onehand",
    "rather", "sarather", "trill", "トリル", "delay", "gb", "obj", "aria",
    "dios", "clover", "gunn", "gungnir", "グングニル", "black", "7", "07",
    "an", "sp", "dp", "bms",
]


def base_title(s: str) -> str:
    """去掉 [..]/(..) 内容与难度后缀，得到基础曲名。"""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[\[（(].*?[\]）)]", "", s)   # 去掉括号内容
    s = re.sub(r"[\W_]+", "", s)
    changed = True
    while changed and s:
        changed = False
        for tok in _DIFF_TOKENS:
            if s.endswith(tok) and len(s) > len(tok):
                s = s[: -len(tok)]
                changed = True
                break
    return s


def artist_key(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    parts = [p for p in s.split("/") if p.strip()]
    s = parts[0] if parts else ""   # 多作者时取第一个非空段
    return re.sub(r"[\W_]+", "", s)


def song_group_key(rec: Dict[str, Any]) -> str:
    # 主键：基础曲名（同曲不同差分、不同 noter 的差分必须同组）。
    # 注意：存在极少数不同歌曲同曲名被合并的风险，由 group_summary 的
    # duration/note 方差标记供人工复核。
    title = base_title(rec.get("title", ""))
    if title:
        return f"t:{title}"
    # 无可靠标题：退化为所在目录（通常一个目录 = 一首歌的包）
    rel = rec.get("rel_path", rec.get("path", ""))
    parts = rel.replace("\\", "/").split("/")
    folder = os.path.dirname(rel).replace("\\", "/")
    folder = re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", folder).lower())
    return f"f:{folder or 'unknown'}"


def group_split(records: List[Dict[str, Any]], val_ratio: float = 0.2,
                seed: int = 0, mode: str = "song") -> Tuple[List[int], List[int], Dict[str, Any]]:
    """按 group 划分。返回 (train_idx, val_idx, report)。

    mode="song"：按 song_group_key 分组（同一曲目差分必须同侧）；
    mode="chart"：每个 chart 独立成组（等价于随机按文件划分，供对照/调试）。
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        gid = song_group_key(rec) if mode == "song" else f"chart:{i}"
        groups[gid].append(i)

    group_ids = sorted(groups.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(group_ids)

    n_val = max(1, int(round(len(group_ids) * val_ratio)))
    val_groups = set(group_ids[:n_val])
    train_idx, val_idx = [], []
    for g in group_ids:
        (val_idx if g in val_groups else train_idx).extend(groups[g])

    sizes = sorted(len(v) for v in groups.values())
    report = {
        "n_charts": len(records),
        "n_groups": len(groups),
        "group_size_min": sizes[0] if sizes else 0,
        "group_size_max": sizes[-1] if sizes else 0,
        "group_size_hist": dict(sorted(Counter(sizes).items())),
        "train_charts": len(train_idx),
        "val_charts": len(val_idx),
        "train_groups": len(group_ids) - len(val_groups),
        "val_groups": len(val_groups),
    }
    return train_idx, val_idx, report


def group_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按 song 分组的概览：组大小、组内 duration/note 范围（用于发现误合并）。"""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[song_group_key(r)].append(r)
    out = {}
    for gid, recs in groups.items():
        durs = [r["meta"]["duration_sec"] for r in recs]
        notes = [r["meta"]["total_notes"] for r in recs]
        out[gid] = {
            "n_charts": len(recs),
            "duration_min": round(min(durs), 1),
            "duration_max": round(max(durs), 1),
            "notes_min": min(notes),
            "notes_max": max(notes),
            "titles": sorted({r["title"] for r in recs})[:5],
        }
    # 标记疑似误合并（组内时长跨度 > 50% 或 note 数差 > 5 倍）
    flagged = []
    for gid, info in out.items():
        if info["n_charts"] < 2:
            continue
        dur_span = (info["duration_max"] - info["duration_min"]) / max(info["duration_min"], 1e-6)
        note_ratio = info["notes_max"] / max(info["notes_min"], 1e-6)
        if dur_span > 0.5 or note_ratio > 5.0:
            flagged.append({**info, "group_id": gid,
                            "dur_span": round(dur_span, 2), "note_ratio": round(note_ratio, 2)})
    return {"groups": out, "flagged_merges": flagged}


def leakage_report(records, train_idx, val_idx) -> Dict[str, int]:
    """检查跨 split 是否出现同 title（按归一化 title）。"""
    t_train = {_norm(records[i].get("title", "")) for i in train_idx}
    t_val = {_norm(records[i].get("title", "")) for i in val_idx}
    both = t_train & t_val
    return {"same_title_cross_split": len(both), "examples": sorted(both)[:10]}

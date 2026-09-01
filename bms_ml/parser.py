"""BMS 文本解析器。

负责把 .bms/.bme/.bml 文本解析成 RawChart（头命令 + 通道对象），
并检测需要隔离的情况（控制流、DP、MGQ-LN 等）。

参考：
- hitkey BMS command memo
- bemusic/bmspec（可执行规范）
- bms-skills 技能包（通道映射 / 格式笔记 / base62）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------- 基础工具 ----------------

def decode_bytes(raw: bytes) -> str:
    """按常见编码依次尝试解码 BMS 文本。"""
    for enc in ("utf-8-sig", "utf-8", "cp932", "gb18030", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def idx_value(chars: str, base: int) -> int:
    """两位索引字符 → 十进制值。base=36（默认）或 62。

    36 进制下大小写等价；62 进制下小写 a-z = 36-61（#BASE 62 语义）。
    """
    v = 0
    for ch in chars:
        if "0" <= ch <= "9":
            d = ord(ch) - ord("0")
        elif "A" <= ch <= "Z":
            d = ord(ch) - ord("A") + 10
        elif "a" <= ch <= "z":
            d = ord(ch) - ord("a") + 10
            if base == 62:
                d += 26
        else:
            return -1
        if d >= base:
            return -1
        v = v * base + d
    return v


def hex_byte_value(chars: str) -> int:
    """通道 03 的两位十六进制 → BPM 值（0-255）。"""
    try:
        return int(chars, 16)
    except ValueError:
        return -1


# ---------------- 数据结构 ----------------


@dataclass
class Issue:
    """解析过程中记录的一个问题。code 用于机器识别，detail 用于人读。"""
    code: str
    detail: str = ""


@dataclass
class ChannelLine:
    measure: int
    channel: str          # 原始两位通道（如 "11"、"16"、"51"）
    raw_values: str       # 冒号后的原始字符串
    line_no: int


@dataclass
class RawChart:
    """解析后的原始谱面：头命令 + 通道行列表 + 问题记录。"""
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    bpm_defs: Dict[int, float] = field(default_factory=dict)   # #BPMxx
    stop_defs: Dict[int, float] = field(default_factory=dict)  # #STOPxx（单位：1/192 小节 = 1/48 拍）
    lnobj: set = field(default_factory=set)                    # #LNOBJ 索引集合
    base: int = 36
    player: int = 1
    rank: int = 2
    lntype: int = 1
    channels: List[ChannelLine] = field(default_factory=list)
    control_flow: List[str] = field(default_factory=list)      # 出现的控制流命令
    issues: List[Issue] = field(default_factory=list)
    encoding: str = "?"


# ---------------- 解析 ----------------

# 控制流命令（出现即不可静态解析 → 隔离）
_CTRL = re.compile(
    r"^\s*#\s*(RANDOM|SETRANDOM|IF|ELSEIF|ELSE|ENDIF|ENDRANDOM|"
    r"SWITCH|SETSWITCH|CASE|SKIP|DEF|ENDSW)\b",
    re.IGNORECASE,
)
# 头命令：#KEY 值 或 #KEYxx 值（key 可含数字索引，如 #STOP11 / #BPM01 / #WAVZZ）
_HEADER = re.compile(r"^\s*#\s*([A-Za-z][0-9A-Za-z]*)\s*(.*)$")
# 通道行：#MMMCC:...（可带 EXT 前缀；小节号至少 3 位，通道两位）
_CHANNEL = re.compile(r"^\s*#\s*(?:EXT\s+)?#?(\d{3,})([0-9A-Za-z]{2})\s*:\s*(.*)$")


def parse_bms_text(text: str, path: str = "?") -> RawChart:
    chart = RawChart(path=path)
    line_no = 0
    for line in text.splitlines():
        line_no += 1
        line = line.rstrip("\r")
        s = line.strip()
        if not s or not s.startswith("#"):
            continue  # 非 # 开头 = 注释

        m = _CTRL.match(s)
        if m:
            chart.control_flow.append(m.group(1).upper())
            continue

        m = _CHANNEL.match(s)
        if m:
            try:
                measure = int(m.group(1))
            except ValueError:
                chart.issues.append(Issue("bad_measure", f"line {line_no}: {line[:60]}"))
                continue
            if measure > 999:
                chart.issues.append(
                    Issue("measure_gt_999", f"measure {measure} at line {line_no}")
                )
            chart.channels.append(
                ChannelLine(measure, m.group(2).upper(), m.group(3), line_no)
            )
            continue

        m = _HEADER.match(s)
        if m:
            key = m.group(1).upper()
            value = m.group(2).strip()
            _handle_header(chart, key, value, line_no)
            continue

        chart.issues.append(Issue("unparsed_line", f"line {line_no}: {line[:60]}"))
    return chart


def _handle_header(chart: RawChart, key: str, value: str, line_no: int) -> None:
    if key in ("TITLE", "SUBTITLE", "ARTIST", "SUBARTIST", "GENRE", "MAKER",
               "COMMENT", "PLAYLEVEL", "TOTAL", "DIFFICULTY", "STAGEFILE",
               "BANNER", "BACKBMP", "CHARSET", "PATH_WAV"):
        chart.headers[key] = value
        return
    if key == "BPM":
        try:
            bpm = float(value)
            chart.headers["BPM"] = str(bpm)
        except ValueError:
            chart.issues.append(Issue("bad_bpm_header", f"line {line_no}: {value}"))
        return
    if key == "RANK":
        try:
            chart.rank = int(float(value))
        except ValueError:
            pass
        return
    if key == "PLAYER":
        try:
            chart.player = int(value.strip())
        except ValueError:
            pass
        return
    if key == "LNTYPE":
        try:
            chart.lntype = int(float(value))
        except ValueError:
            pass
        return
    if key == "BASE":
        if value.strip() == "62":
            chart.base = 62
        return
    if key == "LNOBJ":
        # 可多个 #LNOBJ；值为索引（36/62 进制）
        v = idx_value(value.strip()[:2], chart.base)
        if v >= 0:
            chart.lnobj.add(v)
        return
    if key.startswith("WAV"):
        v = idx_value(key[3:5], chart.base)
        if v >= 0:
            chart.headers[f"WAV:{v}"] = value
        return
    if key.startswith("BPM") and len(key) > 3:
        v = idx_value(key[3:5], chart.base)
        if v >= 0:
            try:
                chart.bpm_defs[v] = float(value)
            except ValueError:
                chart.issues.append(Issue("bad_bpm_def", f"line {line_no}: {value}"))
        return
    if key.startswith("EXBPM") and len(key) > 5:
        v = idx_value(key[5:7], chart.base)
        if v >= 0:
            try:
                chart.bpm_defs[v] = float(value)
            except ValueError:
                pass
        return
    if key.startswith("STOP") and len(key) > 4:
        v = idx_value(key[4:6], chart.base)
        if v >= 0:
            try:
                chart.stop_defs[v] = float(value)
            except ValueError:
                chart.issues.append(Issue("bad_stop_def", f"line {line_no}: {value}"))
        return
    # 其余头命令（BGA/SCROLL/OPTION 等）不参与难度计算，记录存在即可
    chart.headers.setdefault("_seen_headers", "")
    chart.headers["_seen_headers"] += key + ","

"""数据检查：对谱面目录做快速摸底，输出报告（不解析完整时间轴）。

用法: python -m bms_ml.check_data --data-dir <dir>
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import re

from .parser import decode_bytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=r"F:\Projects\bms judge\testbms")
    args = ap.parse_args()

    files = sorted({
        f
        for ext in (".bms", ".bme", ".bml", ".bmx")
        for f in glob.glob(os.path.join(args.data_dir, "**", "*" + ext), recursive=True)
    })
    print(f"chart files: {len(files)}")
    print("by ext:", dict(collections.Counter(os.path.splitext(f)[1].lower() for f in files)))

    ctrl_re = re.compile(
        r"^\s*#\s*(RANDOM|SETRANDOM|IF|ELSEIF|ELSE|ENDIF|ENDRANDOM|"
        r"SWITCH|SETSWITCH|CASE|SKIP|DEF|ENDSW)\b", re.I)
    player_hist = collections.Counter()
    base62 = 0
    ctrl = 0
    lnobj = 0
    enc_hist = collections.Counter()
    for f in files:
        with open(f, "rb") as fh:
            raw = fh.read()
        text = decode_bytes(raw)
        for enc in ("utf-8-sig", "utf-8", "cp932", "gb18030"):
            try:
                raw.decode(enc)
                enc_hist[enc] += 1
                break
            except UnicodeDecodeError:
                continue
        for line in text.splitlines():
            if ctrl_re.match(line):
                ctrl += 1
                break
            if re.match(r"^\s*#\s*BASE\s+62\s*$", line, re.I):
                base62 += 1
            m = re.match(r"^\s*#\s*PLAYER\s+(\d+)", line, re.I)
            if m:
                player_hist[int(m.group(1))] += 1
            if re.match(r"^\s*#\s*LNOBJ\b", line, re.I):
                lnobj += 1
    print("encoding:", dict(enc_hist))
    print(f"control-flow files: {ctrl}")
    print(f"#BASE 62 files: {base62}")
    print("#PLAYER:", dict(sorted(player_hist.items())))
    print(f"files with #LNOBJ: {lnobj}")


if __name__ == "__main__":
    main()

"""下载 BMS 难度表（头部 JSON + 数据 JSON）到 output/tables/。

用法:
  python -m bms_ml.fetch_tables --proxy http://127.0.0.1:7897

表 URL 来自 beatoraja-english-guide 的 Difficulty Tables 页面。
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse

from .labels import fetch_table_json


# 官方源（可用时）
KNOWN_TABLES = {
    "satellite": "https://stellabms.xyz/sl/header.json",
    "stella": "https://stellabms.xyz/st/header.json",
    "insane2": "https://rattoto10.github.io/second_table/insane_header.json",
    "normal2": "https://rattoto10.github.io/second_table/header.json",
}

# 国内镜像（https://zris.work/bmstable.htm），默认使用，访问更稳
MIRROR_TABLES = {
    "normal": "https://zris.work/bmstable/normal/normal_header.json",
    "insane": "https://zris.work/bmstable/insane/insane_header.json",
    "overjoy": "https://zris.work/bmstable/overjoy/header.json",
    "normal2": "https://zris.work/bmstable/normal2/header.json",
    "insane2": "https://zris.work/bmstable/insane2/insane_header.json",
    "satellite": "https://zris.work/bmstable/satellite/header.json",
    "stella": "https://zris.work/bmstable/stella/header.json",
    "ln": "https://zris.work/bmstable/ln/ln_header.json",
}


def resolve(base: str, ref: str) -> str:
    return urllib.parse.urljoin(base, ref)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default="http://127.0.0.1:7897")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "tables"))
    ap.add_argument("--tables", default=",".join(MIRROR_TABLES))
    ap.add_argument("--source", choices=["mirror", "original"], default="mirror")
    args = ap.parse_args()

    sources = MIRROR_TABLES if args.source == "mirror" else KNOWN_TABLES
    os.makedirs(args.out, exist_ok=True)
    for name in [t.strip() for t in args.tables.split(",") if t.strip()]:
        if name not in sources:
            print(f"skip unknown table: {name}")
            continue
        header_url = sources[name]
        hp = os.path.join(args.out, name + "_header.json")
        fetch_table_json(header_url, hp, proxy=args.proxy)
        with open(hp, "r", encoding="utf-8") as f:
            h = json.load(f)
        data_url = resolve(header_url, h.get("data_url", ""))
        dp = os.path.join(args.out, name + "_data.json")
        fetch_table_json(data_url, dp, proxy=args.proxy)
        with open(dp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", [])
        print(f"{name}: header={h.get('name')} symbol={h.get('symbol')} "
              f"entries={len(data)}  data_url={data_url}")


if __name__ == "__main__":
    main()

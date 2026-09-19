"""Step 1/3 of the MinaCalc MSD build: note sequences -> row masks (binary blob).

MMA ships MinaCalc as a WASM whose FFI takes (keycount, row masks u32[], row seconds
f32[], row count) - no BPM, no timing points. Our corpus note sequences are
(time_sec, lane, type_code, duration_sec), so a chart becomes rows by OR-ing the lane
masks of notes that share a millisecond-rounded start time - the same grouping calc.js's
buildRows does for .osu input.

Chart set: the union of the fence (sl/st/insane tables) and every chart any player has
played, because those are exactly the charts the response/dev machinery consumes on both
the training rows and the recommendation candidates.

Binary layout (little endian, repeated per chart):
    sha256       64 bytes ascii
    keycount     u32
    n_rows       u32
    masks        u32 * n_rows   (bit i = lane i has a note start at this row)
    times        f32 * n_rows   (SECONDS - the wasm wants seconds, like calc.js)

LN starts are included as taps and LN ends are ignored, matching calc.js (rows built
from noteStarts only) and Etterna's RC-oriented usage. keycount < 4 is skipped
(MinaCalc supports 4..18); missing sequences are logged and stay NaN downstream.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "bms_ml" / "output" / "phase3" / "dataset"
SEQ = ROOT / "bms_ml" / "output" / "corpus" / "sequences"
OUT = ROOT / "bms_ml" / "output" / "phase3" / "msd"
MIN_KEYCOUNT = 4


def chart_set() -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "bms_ml" / "phase3"))
    from data import load_tables
    tab = load_tables()[["sha256"]]
    played = pd.read_parquet(DS / "firstplays.parquet")[["sha256"]]
    return pd.concat([tab, played]).drop_duplicates("sha256").reset_index(drop=True)


def rows_from_sequence(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    t_ms = np.rint(seq[:, 0].astype(np.float64) * 1000.0).astype(np.int64)
    lane = seq[:, 1].astype(np.int64)
    uniq, inv = np.unique(t_ms, return_inverse=True)
    masks = np.zeros(len(uniq), dtype=np.uint32)
    np.bitwise_or.at(masks, inv, (np.uint32(1) << lane.astype(np.uint32)))
    return masks, uniq.astype(np.float32) / 1000.0, int(lane.max()) + 1


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    charts = chart_set()
    print(f"chart set (fence U played): {len(charts)}")

    blob = bytearray()
    stats = {"ok": 0, "missing_seq": 0, "empty": 0, "too_few_keys": 0, "too_few_rows": 0}
    keycounts = []
    for sha in charts["sha256"].values:
        f = SEQ / f"{sha}.npy"
        if not f.exists():
            stats["missing_seq"] += 1
            continue
        seq = np.load(f)
        if len(seq) == 0:
            stats["empty"] += 1
            continue
        masks, times, keycount = rows_from_sequence(seq)
        if keycount < MIN_KEYCOUNT:
            stats["too_few_keys"] += 1
            continue
        if len(masks) < 2:
            stats["too_few_rows"] += 1
            continue
        blob += sha.encode("ascii")
        blob += struct.pack("<II", keycount, len(masks))
        blob += masks.tobytes()
        blob += times.astype("<f4").tobytes()
        stats["ok"] += 1
        keycounts.append(keycount)

    (OUT / "charts.bin").write_bytes(blob)
    kc = pd.Series(keycounts).value_counts().sort_index()
    print(f"wrote {stats}")
    print("keycount distribution:")
    for k, v in kc.items():
        print(f"  {int(k)} cols: {v:6d} ({100 * v / len(keycounts):.1f}%)")
    print(f"-> {OUT / 'charts.bin'} ({len(blob) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

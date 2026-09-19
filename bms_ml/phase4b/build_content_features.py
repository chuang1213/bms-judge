"""Phase 4B step 0 — build the full-coverage content feature tables.

Three jobs, in this order:

  0.1 SATURATION CHECK (runs first, by user decision 2026-09-12)
      MinaCalc clamps internal skillset values at 40.0, so ultra-hard charts can pile
      up at exactly 40.00 and lose their ordering. Before spending anything, measure
      whether the clamp is actually binding and WHICH difficulty tiers it binds in.
      Decision rule agreed with the user: if the capped charts concentrate in the
      high tiers (sl11-12 / st8+), build a patched copy of the WASM (cap 40 -> 100)
      and recompute; otherwise keep the stock binary.

  0.2 MSD full coverage
      Phase 3 built MSD only for its own fence (table U played) = 6,509 charts, which
      is why `msd.parquet` has 5,497 rows. Phase 4B's cold-chart universe is the
      cross_section SP charts (12,292). Same three-stage pipeline (prep -> node/WASM ->
      finalize), only the input list changes. Measured cost ~1 ms/chart.

  0.3 perm_space full coverage
      Same story: `chart_perm_space.parquet` has 4,262 rows from Phase 3's scope. The
      implementation itself is reused unchanged (`phase3/chart_perm_space.py`), run
      over the wider list. Measured ~0.09 s/chart.

PATCHED AND UNPATCHED MSD ARE NEVER MERGED INTO ONE COLUMN. They land in separate
parquet files with a `msd_cap` provenance tag, and downstream code must pick one.

Missing values stay NaN. Nothing is imputed: filling 0 would make "no information"
look like "very easy", and filling the mean would invent a medium-difficulty chart.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PHASE4 = ROOT / "bms_ml" / "phase4"
PHASE3 = ROOT / "bms_ml" / "phase3"
sys.path.insert(0, str(PHASE4))
sys.path.insert(0, str(PHASE3))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from provenance import header, write_result  # noqa: E402

CROSS_SECTION = ROOT / "bms_ml/output/phase4/dataset/cross_section.parquet"
SEQ_DIR = ROOT / "bms_ml/output/corpus/sequences"
MSD_PHASE3 = ROOT / "bms_ml/output/phase3/dataset/msd.parquet"
PERM_PHASE3 = ROOT / "bms_ml/output/phase3/dataset/chart_perm_space.parquet"
OUT = ROOT / "bms_ml/output/phase4/dataset"
WORK = ROOT / "bms_ml/output/phase4/features_build"

MMA_ETT = (ROOT / "ref_repo/osumania_map_analyser-main"
           "/ManiaMapAnalyser by Leo_Black/js/ett")
WASM_VERSION = "74.0"
CAP_VALUE = 40.00          # MinaCalc's internal clamp
CAP_TOL = 0.005            # "at the cap" tolerance
SKILLSETS = ["Overall", "Stream", "Jumpstream", "Handstream", "Stamina",
             "JackSpeed", "Chordjack", "Technical"]


# --------------------------------------------------------------------------- #
# shared
# --------------------------------------------------------------------------- #
def sp_universe() -> pd.DataFrame:
    """(sha256, notes) for every SP chart Phase 4B can ever score.

    This is the cross_section SP set, NOT the whole 7-key corpus: a cold chart still
    has to appear in some player's history to be an evaluation row.
    """
    xs = pd.read_parquet(CROSS_SECTION, columns=["sha256", "mode_kind"])
    sp = xs.loc[xs["mode_kind"] == "sp", "sha256"].drop_duplicates().reset_index(drop=True)
    return pd.DataFrame({"sha256": sp})


def sequence_present(sha: str) -> bool:
    return (SEQ_DIR / f"{sha}.npy").exists()


def pack_charts(shas: list[str], out_bin: Path) -> dict:
    """note sequences -> the row-mask blob MinaCalc's FFI consumes."""
    from msd_prep import rows_from_sequence

    blob = bytearray()
    stats = {"packed": 0, "missing_sequence": 0, "empty": 0,
             "too_few_keys": 0, "too_few_rows": 0}
    for sha in shas:
        f = SEQ_DIR / f"{sha}.npy"
        if not f.exists():
            stats["missing_sequence"] += 1
            continue
        seq = np.load(f)
        if len(seq) == 0:
            stats["empty"] += 1
            continue
        masks, times, keycount = rows_from_sequence(seq)
        if keycount < 4:
            stats["too_few_keys"] += 1
            continue
        if len(masks) < 1:
            stats["too_few_rows"] += 1
            continue
        blob += sha.encode("ascii")
        blob += struct.pack("<II", keycount, len(masks))
        blob += masks.tobytes()
        blob += times.astype("<f4").tobytes()
        stats["packed"] += 1
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    out_bin.write_bytes(blob)
    stats["blob_mb"] = round(len(blob) / 1e6, 2)
    return stats


def run_minacalc(bin_path: Path, out_jsonl: Path, ett_dir: Path) -> dict:
    """Run the reuse-as-is node driver (phase3/msd_compute.mjs) via a shim.

    `msd_compute.mjs` hard-codes the reference WASM directory, so rather than edit a
    working Phase 3 script we point it at a temp tree that mirrors the layout. That
    keeps the patch/unpatched choice out of Phase 3's code entirely.
    """
    import time
    shim = WORK / "ett_shim"
    (shim / "versions").mkdir(parents=True, exist_ok=True)
    for name in (f"minaclac-{WASM_VERSION}.js", f"minaclac-{WASM_VERSION}.wasm"):
        src = ett_dir / "versions" / name
        if not src.exists():
            raise FileNotFoundError(f"missing reference WASM asset: {src}")
        shutil.copyfile(src, shim / "versions" / name)
    wrapper = WORK / "run_minacalc.mjs"
    wrapper.write_text(
        "import { readFileSync } from 'node:fs';\n"
        "import { pathToFileURL } from 'node:url';\n"
        "import path from 'node:path';\n"
        "const [shimDir, binPath, outPath] = process.argv.slice(2);\n"
        "const V = '0.74.0';\n"
        "const S = ['Overall','Stream','Jumpstream','Handstream','Stamina',"
        "'JackSpeed','Chordjack','Technical'];\n"
        "const glue = pathToFileURL(path.join(shimDir, 'versions', `minaclac-74.0.js`));\n"
        "const { default: createMinaCalc } = await import(glue.href);\n"
        "const wasmBinary = new Uint8Array(readFileSync("
        "path.join(shimDir, 'versions', `minaclac-74.0.wasm`)));\n"
        "const mod = await createMinaCalc({ wasmBinary, locateFile: () => "
        "path.join(shimDir, 'versions', `minaclac-74.0.wasm`) });\n"
        "const buf = readFileSync(binPath);\n"
        "const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);\n"
        "const { openSync, writeSync, closeSync } = await import('node:fs');\n"
        "const fd = openSync(outPath, 'w');\n"
        "const dec = new TextDecoder();\n"
        "let off = 0, ok = 0, fail = 0;\n"
        "while (off < buf.byteLength) {\n"
        "  const sha = dec.decode(buf.subarray(off, off + 64)); off += 64;\n"
        "  const keycount = view.getUint32(off, true); off += 4;\n"
        "  const nRows = view.getUint32(off, true); off += 4;\n"
        "  const masks = new Uint32Array(buf.buffer, buf.byteOffset + off, nRows);"
        " off += 4 * nRows;\n"
        "  const times = new Float32Array(buf.buffer, buf.byteOffset + off, nRows);"
        " off += 4 * nRows;\n"
        "  const mPtr = mod._malloc(nRows * 4), tPtr = mod._malloc(nRows * 4);\n"
        "  const oPtr = mod._malloc(S.length * 4);\n"
        "  try {\n"
        "    mod.HEAPU32.set(masks, mPtr >>> 2); mod.HEAPF32.set(times, tPtr >>> 2);\n"
        "    const good = mod._minacalc_compute(keycount, 1.0, 0.93, mPtr, tPtr,"
        " nRows, oPtr);\n"
        "    if (good) {\n"
        "      const raw = mod.HEAPF32.slice(oPtr >>> 2, (oPtr >>> 2) + S.length);\n"
        "      const values = Object.fromEntries(S.map((k, i) => [k, Number(raw[i]) || 0]));\n"
        "      writeSync(fd, JSON.stringify({ sha256: sha, keycount, values }) + '\\n');"
        " ok++;\n"
        "    } else { fail++; writeSync(fd, JSON.stringify({ sha256: sha,"
        " minacalc_failed: true }) + '\\n'); }\n"
        "  } catch (e) { fail++; }\n"
        "  mod._free(mPtr); mod._free(tPtr); mod._free(oPtr);\n"
        "}\n"
        "closeSync(fd);\n"
        "console.log(`ok ${ok}, fail ${fail}`);\n", encoding="utf-8")
    t0 = time.time()
    r = subprocess.run(["node", str(wrapper), str(shim), str(bin_path), str(out_jsonl)],
                       capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"node MinaCalc run failed: {r.stderr[-2000:]}")
    return {"wall_sec": round(time.time() - t0, 2),
            "node_stdout": (r.stdout or "").strip()[-200:]}


def jsonl_to_frame(jsonl: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Parse the MinaCalc JSONL into (frame, failures).

    Failures are returned rather than dropped: a chart MinaCalc refuses is a real
    coverage gap and has to appear in the report.
    """
    rows, failures = [], []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "values" in r:
                rows.append({"sha256": r["sha256"],
                             **{f"msd_{k.lower()}": r["values"][k] for k in SKILLSETS}})
            elif r.get("minacalc_failed"):
                failures.append({"sha256": r["sha256"], "why": "minacalc returned 0"})
            elif "error" in r:
                failures.append({"sha256": r["sha256"], "why": str(r["error"])[:200]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, failures
    tc = df["msd_technical"]
    if tc.nunique() <= 2:      # constant on the n-key path -> dead column
        df = df.drop(columns=["msd_technical"])
    return df.drop_duplicates("sha256").reset_index(drop=True), failures


def make_patched_wasm() -> Path:
    """Copy the reference WASM aside and patch f32.const 40.0 -> 100.0.

    NEVER writes into ref_repo. Returns the patched tree's ett dir.
    """
    patched = WORK / "ett_patched"
    (patched / "versions").mkdir(parents=True, exist_ok=True)
    for name in (f"minaclac-{WASM_VERSION}.js", f"minaclac-{WASM_VERSION}.wasm"):
        shutil.copyfile(MMA_ETT / "versions" / name, patched / "versions" / name)
    wasm = patched / "versions" / f"minaclac-{WASM_VERSION}.wasm"
    buf = bytearray(wasm.read_bytes())
    old, new = bytes([0x43, 0x00, 0x00, 0x20, 0x42]), bytes([0x43, 0x00, 0x00, 0xC8, 0x42])
    n = 0
    i = 0
    while i + len(old) <= len(buf):
        if bytes(buf[i:i + len(old)]) == old:
            buf[i:i + len(new)] = new
            n += 1
            i += len(old)
        else:
            i += 1
    wasm.write_bytes(bytes(buf))
    print(f"  patched WASM: {n} occurrence(s) of f32.const 40.0 -> 100.0")
    return patched


# --------------------------------------------------------------------------- #
# 0.1 saturation check
# --------------------------------------------------------------------------- #
def msd_axis_columns(df: pd.DataFrame) -> list[str]:
    """The real MSD axes. `msd_cap` is a PROVENANCE tag (it equals the clamp value by
    construction), so treating it as an axis makes every row look saturated."""
    return [c for c in df.columns if c.startswith("msd_") and c != "msd_cap"]


def saturation_report(msd: pd.DataFrame, levels: pd.DataFrame,
                      cap: float = CAP_VALUE) -> dict:
    """Is MinaCalc's skillset clamp binding, and in which tiers?

    `cap` must be the clamp the file was produced with (40.0 stock, 100.0 patched) -
    comparing a patched file against 40.0 marks every row as "capped".
    """
    cols = msd_axis_columns(msd)
    per_col = {}
    for c in cols:
        at_cap = np.isclose(msd[c], cap, atol=CAP_TOL)
        per_col[c] = {"n_at_cap": int(at_cap.sum()),
                      "frac_at_cap": round(float(at_cap.mean()), 4),
                      "max": round(float(msd[c].max()), 3)}
    at_cap_any = np.isclose(msd[cols], cap, atol=CAP_TOL).any(axis=1)

    j = msd.merge(levels, on="sha256", how="left")
    by_level = {}
    if "level" in j.columns:
        j["level_num"] = pd.to_numeric(j["level"], errors="coerce")
        g = j.dropna(subset=["level_num"]).groupby(["table", "level_num"], sort=True)
        for (table, level), grp in g:
            mask = np.isclose(grp[cols], cap, atol=CAP_TOL).any(axis=1)
            if len(grp) < 3:
                continue
            by_level[f"{table}|{level:g}"] = {
                "n": int(len(grp)), "n_capped": int(mask.sum()),
                "frac_capped": round(float(mask.mean()), 3)}
    # is the clamp flattening the top of the difficulty relation?
    overall = "msd_overall"
    rho_all = rho_uncapped = None
    if "level_num" in j.columns:
        from scipy import stats as sps
        s = j.dropna(subset=["level_num", overall])
        if len(s) > 10:
            rho_all = round(float(sps.spearmanr(s[overall], s["level_num"]).statistic), 4)
            un = s[~np.isclose(s[overall], cap, atol=CAP_TOL)]
            if len(un) > 10:
                rho_uncapped = round(
                    float(sps.spearmanr(un[overall], un["level_num"]).statistic), 4)
    tiers_hit = sorted({k for k, v in by_level.items() if v["n_capped"] > 0})
    return {
        "cap_value": cap,
        "n_charts": int(len(msd)),
        "n_charts_capped_any_axis": int(at_cap_any.sum()),
        "frac_charts_capped_any_axis": round(float(at_cap_any.mean()), 4),
        "per_axis": per_col,
        "by_table_level": by_level,
        "tiers_with_caps": tiers_hit,
        "spearman_msd_overall_vs_level_all": rho_all,
        "spearman_msd_overall_vs_level_below_cap": rho_uncapped,
        "capping_is_binding": bool(int(at_cap_any.sum()) >= 20),
        "note": ("patched/unpatched decision rule (user, 2026-09-12): patch only if the "
                 "capped charts concentrate in the high tiers (sl11-12 / st8+)"),
    }


def levels_table() -> pd.DataFrame:
    """sha256 -> table, level (difficulty tables are used ONLY as a coordinate here,
    never as a feature: PROTOCOL sec 3.1)."""
    from data import load_tables
    t = load_tables()
    return t[["sha256", "table", "level"]].drop_duplicates("sha256")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4B step 0: full-coverage features")
    ap.add_argument("--steps", default="sat,msd,perm",
                    help="comma list of: sat, msd, msd-patched, perm, v2, seq")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    uni = sp_universe()
    seq_ok = [s for s in uni["sha256"] if sequence_present(s)]
    print(f"Phase 4B SP universe: {len(uni)} charts, {len(seq_ok)} with a note sequence")

    report: dict = {"sp_universe": int(len(uni)),
                    "sp_universe_with_sequence": int(len(seq_ok))}

    if "sat" in steps:
        print("\n[0.1] saturation check on the existing (Phase 3 scope) MSD")
        old = pd.read_parquet(MSD_PHASE3)
        report["saturation_phase3_scope"] = saturation_report(old, levels_table())
        r = report["saturation_phase3_scope"]
        print(f"  charts={r['n_charts']} capped_any_axis={r['n_charts_capped_any_axis']} "
              f"({r['frac_charts_capped_any_axis']:.2%})")
        for c, v in r["per_axis"].items():
            if v["n_at_cap"]:
                print(f"    {c:<18} at cap: {v['n_at_cap']:>4}  max={v['max']}")
        print(f"  tiers with caps: {r['tiers_with_caps'][:12]}")
        print(f"  spearman(msd_overall, level) all={r['spearman_msd_overall_vs_level_all']} "
              f"below_cap={r['spearman_msd_overall_vs_level_below_cap']}")

    for tag, ett, needs_patch in (("msd", MMA_ETT, False),
                                  ("msd-patched", None, True)):
        if tag not in steps:
            continue
        out_parquet = OUT / ("msd_cap40.parquet" if not needs_patch else "msd_cap100.parquet")
        if out_parquet.exists() and not args.force:
            print(f"\n[{tag}] cached -> {out_parquet.name}")
            cached = pd.read_parquet(out_parquet)
            cap = 100.0 if needs_patch else CAP_VALUE
            report[tag] = {"cached": True, "rows": int(len(cached)), "cap": cap,
                           "axes": msd_axis_columns(cached),
                           "coverage_of_universe": round(len(cached) / max(len(uni), 1), 4),
                           "saturation": saturation_report(cached, levels_table(), cap=cap)}
            continue
        print(f"\n[0.2] MSD ({tag})")
        if needs_patch:
            ett = make_patched_wasm()
        # CRITICAL: the patched build must recompute EVERY chart. Reusing Phase 3's
        # unpatched values for the remainder would put two different value scales in
        # one column - exactly the mixing the user forbade. The unpatched build may
        # reuse Phase 3's rows because they come from the same binary and version.
        if needs_patch:
            todo = list(seq_ok)
            print(f"  patched build: recomputing ALL {len(todo)} charts (no reuse)")
        else:
            prior = (set(pd.read_parquet(MSD_PHASE3)["sha256"])
                     if MSD_PHASE3.exists() else set())
            todo = [s for s in seq_ok if s not in prior]
            print(f"  reuse {len(prior & set(uni['sha256']))} from Phase 3; "
                  f"compute {len(todo)} new")
        blob = WORK / f"charts_{tag}.bin"
        stats = pack_charts(todo, blob)
        print(f"  pack: {stats}")
        report_pack = stats
        jsonl = WORK / f"msd_{tag}.jsonl"
        run = run_minacalc(blob, jsonl, ett)
        print(f"  minacalc: {run}")
        fresh, failures = jsonl_to_frame(jsonl)
        if failures:
            print(f"  minacalc REJECTED {len(failures)} chart(s): "
                  f"{[f['sha256'][:12] for f in failures[:5]]}")
        if needs_patch:
            merged = fresh                      # full recompute: nothing reused
        else:
            merged = pd.concat([pd.read_parquet(MSD_PHASE3), fresh], ignore_index=True) \
                .drop_duplicates("sha256")
        merged = merged[merged["sha256"].isin(set(uni["sha256"]))].reset_index(drop=True)
        # Drop effectively-dead axes AFTER the merge. Phase 3's drop check ran on the
        # freshly computed batch only, so `Technical` slipped through: on the n-key
        # path it takes 4 distinct values over 12k charts (0.18 for almost everything),
        # i.e. a column that looks like a feature and carries nothing.
        msd_cols = [c for c in merged.columns if c.startswith("msd_")]
        dead = [c for c in msd_cols if merged[c].nunique(dropna=True) <= 5]
        if dead:
            print(f"  dropping effectively-dead axes {dead} "
                  f"({ {c: int(merged[c].nunique(dropna=True)) for c in dead} } distinct values)")
        live = [c for c in msd_cols if c not in dead]
        # provenance tag, not a value: the two files are never concatenated downstream
        merged = merged[["sha256"] + live]
        merged["msd_cap"] = 100.0 if needs_patch else CAP_VALUE
        merged.to_parquet(out_parquet, index=False)
        if len(live) != 7:
            raise AssertionError(f"expected 7 live MSD axes, got {len(live)}: {live}")
        msd_cols = live
        msd_cols = [c for c in merged.columns if c.startswith("msd_")]
        print(f"  -> {out_parquet.name}: {len(merged)} rows x {len(msd_cols)} axes")
        report[tag] = {"rows": int(len(merged)), "pack": report_pack, "minacalc": run,
                       "minacalc_failures": failures,
                       "axes": msd_cols, "cap": 100.0 if needs_patch else CAP_VALUE,
                       "coverage_of_universe": round(len(merged) / max(len(uni), 1), 4),
                       "saturation": saturation_report(
                           merged, levels_table(), cap=100.0 if needs_patch else CAP_VALUE)}
        rs = report[tag]["saturation"]
        print(f"  capped_any_axis={rs['n_charts_capped_any_axis']} "
              f"tiers={rs['tiers_with_caps'][:8]}")

    if "perm" in steps:
        out_parquet = OUT / "perm_space_full.parquet"
        if out_parquet.exists() and not args.force:
            print(f"\n[perm] cached -> {out_parquet.name}")
            report["perm"] = {"cached": True, "rows": int(len(pd.read_parquet(out_parquet)))}
        else:
            print("\n[0.3] perm_space full coverage")
            prior = set(pd.read_parquet(PERM_PHASE3)["sha256"])
            from chart_repr import OBJECTIVE_PERM_COLS
            from chart_perm_space import chart_perm_stats
            rows, missing, failed = [], 0, 0
            todo = [s for s in seq_ok if s not in prior]
            print(f"  reuse {len(prior & set(uni['sha256']))} from Phase 3; compute {len(todo)}")
            for i, sha in enumerate(todo):
                f = SEQ_DIR / f"{sha}.npy"
                if not f.exists():
                    missing += 1
                    continue
                st = chart_perm_stats(np.load(f))
                if st is None or (isinstance(st, dict) and
                                  all(v != v for v in st.values())):
                    failed += 1
                    continue
                rows.append({"sha256": sha, **st})
                if (i + 1) % 1000 == 0:
                    print(f"    {i + 1}/{len(todo)}", flush=True)
            fresh = pd.DataFrame(rows)
            merged = pd.concat([pd.read_parquet(PERM_PHASE3), fresh], ignore_index=True)
            merged = merged.drop_duplicates("sha256")
            merged = merged[merged["sha256"].isin(set(uni["sha256"]))].reset_index(drop=True)
            merged.to_parquet(out_parquet, index=False)
            print(f"  -> {out_parquet.name}: {len(merged)} rows x "
                  f"{len([c for c in merged.columns if c.startswith('ps_')])} cols "
                  f"(missing_seq={missing}, failed={failed})")
            report["perm"] = {"rows": int(len(merged)),
                              "cols": list(OBJECTIVE_PERM_COLS),
                              "missing_sequence": missing, "failed": failed,
                              "coverage_of_universe": round(len(merged) / max(len(uni), 1), 4)}

    if "v2" in steps:
        out_parquet = OUT / "chart_stats_v2_full.parquet"
        if out_parquet.exists() and not args.force:
            print(f"\n[v2] cached -> {out_parquet.name}")
            report["v2"] = {"cached": True,
                            "rows": int(len(pd.read_parquet(out_parquet)))}
        else:
            print("\n[0.4] objective v2 full coverage")
            from chart_repr import OBJECTIVE_V2_COLS
            from chart_stats_v2 import chart_v2_stats
            prior_file = ROOT / "bms_ml/output/phase3/dataset/chart_stats_v2.parquet"
            prior = pd.read_parquet(prior_file) if prior_file.exists() else None
            prior_shas = set(prior["sha256"]) if prior is not None else set()
            todo = [s for s in seq_ok if s not in prior_shas]
            print(f"  reuse {len(prior_shas & set(uni['sha256']))} from Phase 3; "
                  f"compute {len(todo)}")
            rows, missing = [], 0
            for i, sha in enumerate(todo):
                f = SEQ_DIR / f"{sha}.npy"
                if not f.exists():
                    missing += 1
                    rows.append({"sha256": sha, **{c: np.nan for c in OBJECTIVE_V2_COLS}})
                    continue
                rows.append({"sha256": sha, **chart_v2_stats(np.load(f))})
                if (i + 1) % 2000 == 0:
                    print(f"    {i + 1}/{len(todo)}", flush=True)
            frames = [pd.DataFrame(rows)]
            if prior is not None:
                frames.insert(0, prior)
            merged = pd.concat(frames, ignore_index=True).drop_duplicates("sha256")
            merged = merged[merged["sha256"].isin(set(uni["sha256"]))].reset_index(drop=True)
            merged.to_parquet(out_parquet, index=False)
            # NOTE: Phase 3's chart_stats_v2 documents NaN -> trial-median imputation
            # downstream. Phase 4B does NOT keep that: missing stays NaN.
            nan_rows = int(merged[OBJECTIVE_V2_COLS[0]].isna().sum())
            print(f"  -> {out_parquet.name}: {len(merged)} rows x "
                  f"{len(OBJECTIVE_V2_COLS)} cols (missing_seq={missing}, "
                  f"NaN rows={nan_rows}; NOT imputed)")
            report["v2"] = {"rows": int(len(merged)), "cols": list(OBJECTIVE_V2_COLS),
                            "missing_sequence": missing, "nan_rows_not_imputed": nan_rows,
                            "coverage_of_universe": round(len(merged) / max(len(uni), 1), 4),
                            "note": "values left NaN; Phase 3's median imputation dropped"}

    if "seq" in steps:
        # Recover note sequences for charts the corpus pipeline never wrote one for.
        # `build_corpus_manifest.py` saves `sequences/<sha>.npy` only when a chart has
        # NO quarantine tag, so every quarantined chart lacks one - even though its
        # source file parses fine and a player may have a best score on it. Those
        # charts are legitimate cold-chart rows, so MSD/perm/v2 should be computable.
        #
        # This only ADDS files under the shared corpus sequences dir; it never
        # overwrites an existing sequence.
        print("\n[0.5] recover missing note sequences")
        SEQ_DIR.mkdir(parents=True, exist_ok=True)
        have = {p.stem for p in SEQ_DIR.glob("*.npy")}
        missing = [s for s in uni["sha256"] if s not in have]
        print(f"  {len(missing)} SP charts have no sequence")
        manifest = {}
        with open(ROOT / "bms_ml/output/corpus/manifest.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                manifest[r["sha256"]] = r
        # `features`/`parser`/`timeline` are relative-import modules of the bms_ml
        # package, so they must be imported through the package (repo root on path).
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from bms_ml.features import build_note_sequence
        from bms_ml.parser import decode_bytes, parse_bms_text
        from bms_ml.timeline import build_timeline
        wrote, failed, quotients = 0, [], []
        for sha in missing:
            rec = manifest.get(sha)
            if rec is None:
                failed.append({"sha256": sha, "why": "not in manifest"})
                continue
            path = Path(rec["path"])
            if not path.exists():
                failed.append({"sha256": sha, "why": "source file missing",
                               "path": str(path)})
                continue
            try:
                with open(path, "rb") as f:
                    uc = build_timeline(parse_bms_text(decode_bytes(f.read()), str(path)))
                seq = build_note_sequence(uc)
                if len(seq) == 0:
                    failed.append({"sha256": sha, "why": "parsed to zero notes",
                                   "path": str(path),
                                   "quarantine": rec.get("quarantine")})
                    continue
                np.save(SEQ_DIR / f"{sha}.npy", seq)
                wrote += 1
                quotients.extend(rec.get("quarantine") or [])
            except Exception as exc:  # noqa: BLE001 - report, never abort the build
                failed.append({"sha256": sha, "why": f"{type(exc).__name__}: {exc}",
                               "path": str(path), "quarantine": rec.get("quarantine")})
        from collections import Counter as _Counter
        report["sequence_recovery"] = {
            "requested": len(missing), "written": wrote, "failed": len(failed),
            "failures": failed,
            "quarantines_of_recovered": dict(_Counter(quotients).most_common()),
            "note": "only ADDS missing sequences; existing files are never overwritten",
        }
        print(f"  wrote {wrote} new sequences; {len(failed)} failed")
        for f_ in failed[:10]:
            print(f"    FAIL {f_['sha256'][:12]} {f_['why']} {f_.get('quarantine')}")
        now = [s for s in uni["sha256"] if sequence_present(s)]
        report["sp_universe_with_sequence"] = int(len(now))
        print(f"  SP charts with a sequence now: {len(now)}/{len(uni)}")

    write_result("features_build.json",
                 header(script="build_content_features.py",
                        data_config={"cross_section": str(CROSS_SECTION.relative_to(ROOT)),
                                     "steps": steps,
                                     "msd_reference": str(MMA_ETT.relative_to(ROOT)),
                                     "wasm_version": WASM_VERSION,
                                     "no_imputation": True,
                                     "difficulty_table_used_only_as_coordinate": True}),
                 report)

    # ---- cross-version comparison: how much does the patch actually move?
    p40 = OUT / "msd_cap40.parquet"
    p100 = OUT / "msd_cap100.parquet"
    if p40.exists() and p100.exists():
        a, b = pd.read_parquet(p40), pd.read_parquet(p100)
        m = a.merge(b, on="sha256", suffixes=("_cap40", "_cap100"))
        cmp = {"n_common": int(len(m))}
        for c in [c for c in a.columns if c.startswith("msd_")]:
            da, db = m[f"{c}_cap40"], m[f"{c}_cap100"]
            changed = ~np.isclose(da, db, atol=1e-4)
            cmp[c] = {"n_changed": int(changed.sum()),
                      "max_abs_delta": round(float((db - da).abs().max()), 3),
                      "max_cap40": round(float(da.max()), 2),
                      "max_cap100": round(float(db.max()), 2)}
        report["cap40_vs_cap100"] = cmp
        print("\ncap40 vs cap100 (patched) on the common charts:")
        for c, v in cmp.items():
            if isinstance(v, dict) and v["n_changed"]:
                print(f"  {c:<18} changed={v['n_changed']:>5} "
                      f"max|c40={v['max_cap40']} max|c100={v['max_cap100']} "
                      f"max_delta={v['max_abs_delta']}")
        write_result("features_build.json",
                     header(script="build_content_features.py",
                            data_config={"cross_section": str(CROSS_SECTION.relative_to(ROOT)),
                                         "steps": steps,
                                         "msd_reference": str(MMA_ETT.relative_to(ROOT)),
                                         "wasm_version": WASM_VERSION,
                                         "no_imputation": True,
                                         "difficulty_table_used_only_as_coordinate": True}),
                     report)
    print(f"\nwrote {OUT.name}/*.parquet and output/phase4/features_build.json")


if __name__ == "__main__":
    main()

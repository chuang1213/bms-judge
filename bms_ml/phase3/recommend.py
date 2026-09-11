"""Phase 3.6 -> first deliverable: a chart recommender for ONE player.

    python bms_ml/phase3/recommend.py --player reiaki [--top 25]

What it is
----------
The protocol model (the exact feature set of response_dev.B_resp+bpresp+dev, the acc
best at 6.117) is retrained on the train band exactly as the evaluation does, then used
to score every fence chart (sl/st/insane union) the player has NOT played yet.

What it is NOT
--------------
NOT a "practising this will make you stronger" model: the project has no longitudinal /
intervention data, so training value cannot be measured (PHASE3_3_READINESS §6). The
score only means "what the model expects the FIRST play to look like". The grouping
turns that into a usable reading. Thresholds follow the lamp head's ACTUAL spread, not
the nominal ladder: the lamp regression is fitted on an ordinal with 0.5% FC examples,
so its predictions compress toward the middle (measured on vsoflan: q05 1.7, median 4.1,
q95 6.7, max 7.1) and a "predicted FC" group would simply never fire.

  暂缓区  pred_lamp_raw <  3.5   expected fail
  挑战区  3.5 .. 6.0             plausible pass, needs a real attempt
  冲刺区  >= 6.0                 expected clean pass, near the model's ceiling

How candidate features are built (the part worth reading)
---------------------------------------------------------
Candidates are appended to the player's chronological frame as FUTURE rows with NaN
targets, then the *unmodified* protocol machinery runs on the extended frame:

  build_history_features -> the history features, because every cumulative window simply
                            sees the real plays and stops;
  build_table            -> the acc/lamp/bp response blocks AND h_dev_*, because the OLS
                            window is `y_valid & x_valid`, so NaN-target candidates never
                            enter each other's windows.

Zero re-implementation of the causal logic, so deployment features cannot drift from
training features. The one exception is h_knn_acc, which build_history_features would
pollute (a candidate's k-NN window may contain another candidate with NaN acc), so it is
recomputed here: the KNN_K nearest of the player's REAL plays in the same standardised
stat space, same scaler, same k.

Scoring time is "the player's last recorded session" (newest scorelog row), not
wall-clock now: a wall-clock gap of months would push h_days_since_active far beyond the
training distribution - the exact few-shot trap PHASE3_4_TRANSFER documented.

Outputs: bms_ml/output/phase3/recommend/<player>.{html,csv}
"""
from __future__ import annotations

import argparse
import html
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from chart_repr import BEST_FEATURES, HISTORY_FEATURES, MSD_AXES, OBJECTIVE_STAT_COLS
from common import HGBModel, HGB_KW, load_samples, mae
from data import (HISTORY_USES_OFFTABLE, PLAYERS, ROOT, build_history_features,
                  load_firstplays, load_manifest, load_tables)
from response_decay import MIN_HISTORY
from response_features import axis_sd, build_table
from uncertainty_eval import QHead

OUT = ROOT / "bms_ml" / "output" / "phase3" / "recommend"
DS = ROOT / "bms_ml" / "output" / "phase3" / "dataset"
FEATS = BEST_FEATURES                     # the fused best configuration (5.887)
HALF_LIFE = 180.0                         # must match history_response.py's default
TABLE_SYM = {"satellite": "sl", "stella": "st", "insane": "発狂", "normal": "☆"}
KNN_K = 20                                # data.build_history_features default
LAMP_NAME = {1: "FAILED", 2: "ASSIST EZ", 3: "LIGHT ASSIST EZ", 4: "EASY", 5: "CLEAR",
             6: "HARD", 7: "EX HARD", 8: "FULL COMBO", 9: "PERFECT"}


def load_log_counts() -> pd.DataFrame:
    """All scorelog rows per player (activity features use the full row stream)."""
    parts = []
    for player, rel in PLAYERS.items():
        con = sqlite3.connect(ROOT / rel / "scorelog.db")
        d = pd.read_sql_query("SELECT date FROM scorelog", con)
        con.close()
        parts.append(pd.DataFrame({"player": player,
                                   "time": pd.to_datetime(d["date"], unit="s")}))
    return pd.concat(parts, ignore_index=True)


def build_player_frame() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """The protocol frame: same cleaning, same scaler, same Z as data.py main()."""
    man = load_manifest()
    # the four v2 response axes live in a side table keyed on sha256, not in the
    # manifest - without them build_table cannot even index the axis columns; the MSD
    # skillsets come from dataset/msd.parquet the same way (charts the WASM never saw
    # keep NaN and their MSD-block features impute downstream)
    from chart_repr import RESPONSE_AXES
    v2 = pd.read_parquet(ROOT / "bms_ml" / "output" / "phase3" / "dataset"
                         / "chart_stats_v2.parquet")
    v2cols = [c for c in set(RESPONSE_AXES.values()) if c in v2.columns]
    man = man.merge(v2[["sha256"] + v2cols], on="sha256", how="left")
    man = man.merge(pd.read_parquet(ROOT / "bms_ml" / "output" / "phase3" / "dataset"
                                    / "msd.parquet"), on="sha256", how="left")
    tab = load_tables()
    fp = (load_firstplays()
          .merge(man, on="sha256", how="left")
          .merge(tab, on="sha256", how="left"))
    assert not fp.duplicated(["player", "sha256"]).any(), \
        "duplicate (player, sha256): the fence is not 1:1 on sha256"
    fp["acc"] = np.where(fp["notes"] > 0, fp["ex"] * 50.0 / fp["notes"], np.nan)
    fp = fp[fp["bp"] <= fp["notes"] + 5].copy()
    fp = fp[fp["notes"].notna() & (fp["notes"] > 0)].copy()
    fp["is_target"] = fp["table"].notna()
    if not HISTORY_USES_OFFTABLE:
        fp = fp[fp["is_target"]].copy()
    fp = fp.reset_index(drop=True)
    fp["bp_ratio"] = fp["bp"] / fp["notes"]
    stat_cols = [c for c in fp.columns if c.startswith("c_")]
    assert set(stat_cols) == set(OBJECTIVE_STAT_COLS), "chart stat drift vs the registry"
    scaler = StandardScaler().fit(fp[stat_cols].values)
    fp_sorted = fp.sort_values(["player", "time"]).reset_index(drop=True)
    Z = np.nan_to_num(scaler.transform(fp_sorted[stat_cols].values)).astype(np.float64)
    # the response blocks' slope scaling must match history_response.py exactly: it
    # computes its sd on the FULL first-play frame (before any filtering), so do the same
    fp_full = (pd.read_parquet(DS / "firstplays.parquet")
               .merge(v2[["sha256"] + v2cols], on="sha256", how="left")
               .merge(pd.read_parquet(DS / "msd.parquet"), on="sha256", how="left")
               .sort_values(["player", "time"], kind="stable").reset_index(drop=True))
    sd_struct = axis_sd(fp_full)
    sd_msd = axis_sd(fp_full, MSD_AXES)
    return (fp_sorted, man, tab, stat_cols, scaler, Z, sd_struct, sd_msd)


def build_candidates(player: str, fp_sorted: pd.DataFrame, man: pd.DataFrame,
                     tab: pd.DataFrame, stat_cols: list[str], scaler: StandardScaler,
                     Z: np.ndarray, sd: dict, sd_msd: dict,
                     top_log) -> tuple[pd.DataFrame, dict]:
    own = fp_sorted[fp_sorted["player"] == player].reset_index(drop=True)
    if not len(own):
        raise SystemExit(f"player '{player}' has no usable archive rows")
    cand = man.merge(tab, on="sha256", how="inner")
    cand = cand[cand["notes"].notna() & (cand["notes"] > 0)]
    cand = cand[~cand["sha256"].isin(set(own["sha256"]))].reset_index(drop=True)

    # append as future rows; NaN targets keep them out of every causal window
    cext = cand.reindex(columns=own.columns)
    cext["player"] = player
    cext["time"] = top_log
    for c in ("acc", "lamp", "ex", "bp", "bp_ratio", "log1p_bp"):
        cext[c] = np.nan
    ext = pd.concat([own, cext], ignore_index=True)

    Z_own = Z[(fp_sorted["player"] == player).values]
    Z_cand = np.nan_to_num(scaler.transform(cand[stat_cols].values)).astype(np.float64)
    log_counts = load_log_counts()
    H = build_history_features(ext, log_counts, np.vstack([Z_own, Z_cand]))
    hist = H.iloc[len(own):].reset_index(drop=True)

    # h_knn_acc must be recomputed: for every candidate after the first, the built-in
    # window would contain another candidate whose acc is NaN, killing the mean
    acc_own = own["acc"].values.astype(np.float64)
    knn = np.empty(len(cand))
    for i in range(len(cand)):
        d = np.sqrt(((Z_own - Z_cand[i]) ** 2).sum(1))
        kk = min(KNN_K, len(d))
        knn[i] = float(np.mean(acc_own[np.argpartition(d, kk - 1)[:kk]]))
    hist["h_knn_acc"] = knn

    blocks = [
        build_table(ext, sd, min_n=MIN_HISTORY, target="acc", prefix="h_",
                    clip=(0.0, 100.0), half_life=HALF_LIFE),
        build_table(ext, sd, min_n=MIN_HISTORY, target="lamp", prefix="l_",
                    clip=(1.0, 9.0), half_life=HALF_LIFE),
        build_table(ext, sd, min_n=MIN_HISTORY, target="log1p_bp", prefix="b_",
                    clip=(0.0, 12.0), half_life=HALF_LIFE),
        # the MinaCalc axis family (same axes/clips as history_response.py)
        build_table(ext, sd_msd, min_n=MIN_HISTORY, target="acc", prefix="m_",
                    clip=(0.0, 100.0), half_life=HALF_LIFE, axes=MSD_AXES),
        build_table(ext, sd_msd, min_n=MIN_HISTORY, target="lamp", prefix="ml_",
                    clip=(1.0, 9.0), half_life=HALF_LIFE, axes=MSD_AXES),
        build_table(ext, sd_msd, min_n=MIN_HISTORY, target="log1p_bp", prefix="mb_",
                    clip=(0.0, 12.0), half_life=HALF_LIFE, axes=MSD_AXES),
    ]
    feats = pd.concat([hist] + [b.iloc[len(own):].reset_index(drop=True) for b in blocks],
                      axis=1)
    out = pd.concat([cand, feats], axis=1)
    out["has_msd"] = out["msd_overall"].notna()
    meta = {"n_archive": int(len(own)), "n_candidates": int(len(out)),
            "n_no_msd": int((~out["has_msd"]).sum()),
            "as_of": str(top_log)[:19],
            "acc_mean": float(np.nanmean(own["acc"].values)),
            "acc_last20": float(np.nanmean(own["acc"].values[-20:]))}
    return out[["sha256", "title", "artist", "table", "level", "has_msd"] + FEATS], meta


def build_uncertainty(tr: pd.DataFrame, df: pd.DataFrame, cand: pd.DataFrame
                      ) -> dict[str, np.ndarray]:
    """Feature-conditioned bands and pass probability for the candidates.

    Quantile-HGB heads at q10/q50/q90 per target, conformalised (CQR): the correction Q
    is the 80th percentile of the calibration nonconformity, fitted on a per-player time
    split of the train band, never on the test band or the candidates. The variance
    diagnostic showed per-player error is mostly that player's own behavioural variance
    (corr +0.882), and the measured band widths track it (+0.846: reiaki ~10pp wide,
    yangtao ~50pp) - the point prediction alone hides exactly that.
    """
    fit_idx, cal_idx = [], []
    for _p, g in tr.groupby("player"):
        g = g.sort_values("time")
        k = max(1, int(len(g) * 0.8))
        fit_idx.extend(g.index[:k])
        cal_idx.extend(g.index[k:])
    fit, cal = tr.loc[fit_idx], tr.loc[cal_idx]

    def conformal(heads: dict, y_cal: np.ndarray, frame: pd.DataFrame) -> float:
        s = np.maximum(heads[0.1].predict(frame) - y_cal,
                       y_cal - heads[0.9].predict(frame))
        qq = min(1.0, np.ceil((len(s) + 1) * 0.8) / len(s))
        return float(np.quantile(s, qq))

    acc_q = {q: QHead(fit, BEST_FEATURES, fit["acc"].values, q)
             for q in (0.1, 0.5, 0.9)}
    qa = conformal(acc_q, cal["acc"].values, cal)
    out = {
        "pred_acc_lo": acc_q[0.1].predict(cand) - qa,
        "pred_acc_hi": acc_q[0.9].predict(cand) + qa,
    }
    bp_q = {q: QHead(fit, BEST_FEATURES, np.log1p(fit["bp"].values), q)
            for q in (0.1, 0.9)}
    qb = conformal(bp_q, np.log1p(cal["bp"].values), cal)
    blo = bp_q[0.1].predict(cand) - qb
    bhi = bp_q[0.9].predict(cand) + qb
    out["pred_bp_lo"] = np.expm1(np.clip(blo, 0, 12))
    out["pred_bp_hi"] = np.expm1(np.clip(bhi, 0, 12))

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    imp = SimpleImputer(strategy="median").fit(fit[BEST_FEATURES])
    sc = StandardScaler().fit(imp.transform(fit[BEST_FEATURES]))
    clf = HistGradientBoostingClassifier(random_state=0, **HGB_KW)
    clf.fit(sc.transform(imp.transform(fit[BEST_FEATURES])),
            (fit["lamp"].values >= 4).astype(int))
    out["p_pass"] = clf.predict_proba(sc.transform(imp.transform(cand[BEST_FEATURES])))[:, 1]
    return out


def expected_mae(df: pd.DataFrame, models: dict, player: str) -> float | None:
    """Out-of-sample accuracy for THIS player: their test rows were never trained on."""
    te = df[(df["player"] == player) & (df["phase"] == "test")]
    return float(mae(te["acc"], models["acc"].predict(te))) if len(te) else None


def render_html(player: str, cand: pd.DataFrame, meta: dict, pmae: float | None,
                top: int) -> str:
    e = html.escape

    def rows(sub: pd.DataFrame, limit: int) -> str:
        out = []
        for _, r in sub.head(limit).iterrows():
            out.append(
                f"<tr><td class='lv'>{TABLE_SYM.get(r['table'], r['table'])}"
                f"{r['level']:.0f}</td><td class='ti'>{e(str(r['title']))}</td>"
                f"<td class='ar'>{e(str(r['artist'])[:40])}</td>"
                f"<td class='lp n{r['pred_lamp']}'>{r['pred_lamp']} "
                f"{e(r['pred_lamp_name'])}</td>"
                f"<td class='pp'>{100 * r['p_pass']:.0f}%</td>"
                f"<td class='ac'>{r['pred_acc']:.1f} "
                f"<span class='band'>[{r['pred_acc_lo']:.0f}-{r['pred_acc_hi']:.0f}]"
                f"</span></td>"
                f"<td class='bp'>{r['pred_bp']:.0f}</td></tr>")
        return "\n".join(out)

    desc = {"挑战区": "预测能过、但要认真打 —— 主体推荐段",
            "冲刺区": "预测能顺利通过、接近模型上限 —— 冲高难 / 收歌",
            "暂缓区": "预测过不了 —— 先放一放"}
    groups = []
    for g in ("挑战区", "冲刺区", "暂缓区"):
        sub = cand[cand["group"] == g]
        groups.append(
            f"<section><h2>{g} <span class='cnt'>{len(sub)} 张</span></h2>"
            f"<p class='desc'>{desc[g]}</p><table><thead><tr><th>表</th><th>标题</th>"
            f"<th>作者</th><th>预测灯</th><th>通过概率</th><th>预测acc (80%区间)"
            f"</th><th>预测BP</th></tr></thead>"
            f"<tbody>{rows(sub, top)}</tbody></table></section>")
    mae_txt = f"{pmae:.2f} 分" if pmae is not None else "该玩家没有测试段样本"
    css = (":root{--bg:#0f1115;--card:#171a21;--line:#262b36;--fg:#e6e9ef;"
           "--dim:#8b93a5;--acc:#5eead4;--warn:#fbbf24;--bad:#f87171}"
           "*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);"
           "font:14px/1.6 'Segoe UI','Microsoft YaHei',sans-serif;padding:32px}"
           ".wrap{max-width:1080px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}"
           ".sub{color:var(--dim);margin:0 0 24px}"
           ".stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}"
           ".stat{background:var(--card);border:1px solid var(--line);border-radius:10px;"
           "padding:12px 18px;min-width:150px}.stat b{display:block;font-size:20px;"
           "color:var(--acc)}.stat span{color:var(--dim);font-size:12px}"
           "h2{font-size:17px;margin:28px 0 2px}.cnt{color:var(--dim);font-size:13px;"
           "font-weight:normal}.desc{color:var(--dim);margin:0 0 8px;font-size:13px}"
           "table{width:100%;border-collapse:collapse;background:var(--card);"
           "border:1px solid var(--line);border-radius:10px;overflow:hidden}"
           "th,td{padding:7px 12px;text-align:left;border-top:1px solid var(--line)}"
           "th{color:var(--dim);font-weight:600;font-size:12px;background:#1b1f29}"
           "td.pp{font-weight:600;color:#a78bfa}"
           ".band{color:var(--dim);font-size:11px}"
           "td.lv{color:var(--acc);font-weight:600;white-space:nowrap}"
           "td.ti{max-width:420px;overflow:hidden;text-overflow:ellipsis;"
           "white-space:nowrap}td.ar{color:var(--dim);max-width:180px;overflow:hidden;"
           "text-overflow:ellipsis;white-space:nowrap}td.ac{font-weight:600}"
           "td.bp{color:var(--dim)}.lp{font-weight:600}"
           ".n1,.n2,.n3{color:var(--bad)}.n4,.n5{color:var(--warn)}"
           ".n6,.n7{color:var(--acc)}.n8,.n9{color:#a78bfa}"
           ".foot{color:var(--dim);font-size:12px;margin-top:28px;line-height:1.8}")
    return (f"<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
            f"<title>BMS 推荐 · {e(player)}</title><style>{css}</style></head>"
            f"<body><div class='wrap'><h1>BMS 谱面推荐 · {e(player)}</h1>"
            f"<p class='sub'>生成于 {datetime.now():%Y-%m-%d %H:%M} · "
            f"打分时点 = 最近一次记录的游玩（{e(meta['as_of'])}）</p>"
            f"<div class='stats'>"
            f"<div class='stat'><b>{meta['n_archive']}</b><span>已打过的谱面</span></div>"
            f"<div class='stat'><b>{meta['n_candidates']}</b><span>表内未打谱面</span></div>"
            f"<div class='stat'><b>{meta['acc_mean']:.1f} / {meta['acc_last20']:.1f}</b>"
            f"<span>历史 acc 均值 / 最近 20 次</span></div>"
            f"<div class='stat'><b>{mae_txt}</b>"
            f"<span>模型对你的预测误差（测试段）</span></div>"
            f"<div class='stat'><b>{meta['n_no_msd']}</b>"
            f"<span>候选缺 MSD（序列未构建，特征插补）</span></div></div>"
            f"{''.join(groups)}"
            f"<div class='foot'>预测含义：若现在第一次打这张谱的期望表现"
            f"（acc 0-100 / lamp 1-9 / BP 残数）。<br>"
            f"这<b>不是</b>\"练了会变强\"的模型 —— 项目没有纵向干预数据，"
            f"训练价值无法度量；分组只是把\"期望首打表现\"翻译成可读的选择。"
            f"完整候选清单（含全部预测值）见同目录 CSV。"
            f"特征与训练完全同源（响应块 180d 半衰期 + 玩家相对偏差），"
            f"协议口径 acc MAE 6.117。</div></div></body></html>")


def main() -> None:
    try:      # the Windows console is often cp936 and mangles the Chinese group names
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="score unplayed fence charts for one player")
    ap.add_argument("--player", required=True, choices=sorted(PLAYERS))
    ap.add_argument("--top", type=int, default=25, help="rows shown per group in the HTML")
    args = ap.parse_args()
    player = args.player

    print("training the protocol model on the train band ...")
    # the fused features come from the dataset artifact: samples.parquet carries the
    # chart stats and the hand-crafted history, history_response.parquet (built at the
    # same 180d half-life) carries every response/dev block, MSD family included
    hr = pd.read_parquet(DS / "history_response.parquet")
    static = OBJECTIVE_STAT_COLS + HISTORY_FEATURES
    need = [c for c in FEATS if c not in static]
    df = load_samples().merge(hr[["player", "sha256"] + [c for c in need
                                         if c in hr.columns]],
                              on=["player", "sha256"], how="left")
    missing = [c for c in FEATS if c not in df.columns]
    if missing:
        raise SystemExit(f"history_response.parquet lacks {len(missing)} columns "
                         f"(e.g. {missing[:3]}); run history_response.py")
    tr = df[df["phase"] == "train"]
    models = {"acc": HGBModel(tr, FEATS, tr["acc"].values),
              "lamp": HGBModel(tr, FEATS, tr["lamp"].values.astype(float)),
              "bp": HGBModel(tr, FEATS, np.log1p(tr["bp"].values))}
    pmae = expected_mae(df, models, player)

    print(f"building candidates for {player} ...")
    (fp_sorted, man, tab, stat_cols, scaler, Z, sd_struct,
     sd_msd) = build_player_frame()
    log_counts = load_log_counts()
    top_log = log_counts.loc[log_counts["player"] == player, "time"].max()
    cand, meta = build_candidates(player, fp_sorted, man, tab, stat_cols, scaler, Z,
                                  sd_struct, sd_msd, top_log)

    cand["pred_acc"] = models["acc"].predict(cand)
    cand["pred_lamp_raw"] = models["lamp"].predict(cand)
    cand["pred_lamp"] = np.clip(np.round(cand["pred_lamp_raw"]), 1, 9).astype(int)
    cand["pred_lamp_name"] = cand["pred_lamp"].map(LAMP_NAME)
    bp_cap = float(np.log1p(df["bp"].max()))
    cand["pred_bp"] = np.expm1(np.clip(models["bp"].predict(cand), 0, bp_cap))
    print("building uncertainty bands ...")
    for k, v in build_uncertainty(tr, df, cand).items():
        cand[k] = v
    cand["group"] = np.where(cand["pred_lamp_raw"] >= 6.0, "冲刺区",
                             np.where(cand["pred_lamp_raw"] >= 3.5, "挑战区", "暂缓区"))
    order = {"挑战区": 0, "冲刺区": 1, "暂缓区": 2}
    cand["_o"] = cand["group"].map(order)
    cand = cand.sort_values(["_o", "pred_lamp_raw", "pred_acc"],
                            ascending=[True, False, False]).reset_index(drop=True)

    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["group", "table", "level", "title", "artist", "pred_lamp",
            "pred_lamp_name", "p_pass", "pred_acc", "pred_acc_lo", "pred_acc_hi",
            "pred_bp", "pred_bp_lo", "pred_bp_hi", "pred_lamp_raw", "has_msd",
            "sha256"]
    csv_path = OUT / f"{player}.csv"
    cand[keep].to_csv(csv_path, index=False, encoding="utf-8-sig")
    html_path = OUT / f"{player}.html"
    html_path.write_text(render_html(player, cand, meta, pmae, args.top), encoding="utf-8")

    print(f"archive {meta['n_archive']} | candidates {meta['n_candidates']} "
          f"| as of {meta['as_of']} | your out-of-sample acc MAE "
          f"{pmae if pmae is not None else float('nan'):.2f}")
    for g in ("挑战区", "冲刺区", "暂缓区"):
        sub = cand[cand["group"] == g]
        print(f"\n== {g}  ({len(sub)} charts) ==")
        for _, r in sub.head(8).iterrows():
            print(f"  {TABLE_SYM.get(r['table'], r['table']):>3}{r['level']:<4.0f}"
                  f" lamp~{r['pred_lamp']}({r['pred_lamp_name']:<10})"
                  f" acc~{r['pred_acc']:5.1f} bp~{r['pred_bp']:6.1f}  {str(r['title'])[:44]}")
    print(f"\nsaved -> {csv_path}\nsaved -> {html_path}")


if __name__ == "__main__":
    main()

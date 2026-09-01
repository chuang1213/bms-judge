"""baseline 训练入口（标签门控）。

用法（有标签时）:
  python -m bms_ml.train --manifest output/manifest.jsonl --table <表名> --out-dir output/runs/run1

没有可靠难度标签时，训练不会开始（会明确提示），
用 --smoke-test 只验证"数据 → 模型 → loss → backward → step"机制是否通畅，
smoke 结果不构成任何实验结果。

标签约定：单表内部把等级视为有序数值做回归（SmoothL1），
不跨表合并，不把 ☆12/★1/★★1 当作同一标尺。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import ChartDataset
from .model import MLPBaseline
from .split import group_split, leakage_report


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "output", "manifest.jsonl"))
    ap.add_argument("--table", default="", help="使用哪个难度表（manifest 中的 labels.table）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--split-mode", choices=["song", "chart"], default="song",
                    help="song=同曲分组划分（默认）；chart=按文件随机划分（对照/调试）")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "output", "runs", "run1"))
    ap.add_argument("--smoke-test", action="store_true",
                    help="用随机标签验证训练机制（不构成实验结果）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  torch {torch.__version__}")

    records = [json.loads(line) for line in open(args.manifest, encoding="utf-8")]
    clean = [r for r in records if not r["quarantine"]]
    print(f"manifest: {len(records)} charts, clean: {len(clean)}")

    # ---- 标签门控 ----
    if args.smoke_test:
        print("\n[SMOKE TEST] 用随机标签验证训练机制，不报告任何有效指标。")
        rng = np.random.RandomState(args.seed)
        for r in clean:
            r["label"] = float(rng.uniform(1.0, 12.0))
        labeled = clean
    else:
        if not args.table:
            print("\n未提供 --table。当前没有可靠难度标签，按项目约定：")
            print("不在伪标签上训练、不报告虚假准确率。")
            print("请先获得难度表数据（labels.py 支持下载/解析），再指定 --table 训练。")
            return
        labeled = []
        for r in clean:
            lab = [l for l in r["labels"] if l["table"] == args.table and l["value"] is not None]
            if lab:
                r["label"] = lab[0]["value"]
                r["level_str"] = lab[0]["level"]
                labeled.append(r)
        print(f"table '{args.table}': labeled charts = {len(labeled)} / {len(clean)}")
        if len(labeled) < 10:
            print("标签覆盖太少（<10），不足以训练，停止。")
            return
        levels = sorted({r["label"] for r in labeled})
        print(f"label levels present: {levels}")
        print("注：单表等级按有序数值回归处理（简化假设，后续可讨论分类/ordinal）。")

    # ---- 同曲分组划分 ----
    train_idx, val_idx, split_report = group_split(
        labeled, args.val_ratio, args.seed, mode=args.split_mode)
    leak = leakage_report(labeled, train_idx, val_idx)
    print(f"split: train={len(train_idx)} val={len(val_idx)} groups={split_report['n_groups']}")
    print(f"leak check (same normalized title cross split): {leak['same_title_cross_split']}")

    train_records = [labeled[i] for i in train_idx]
    val_records = [labeled[i] for i in val_idx]

    # ---- 特征归一化（只在 train 上 fit）----
    X_all = np.stack([np.asarray(r["features"], dtype=np.float32) for r in labeled])
    y_all = np.asarray([r["label"] for r in labeled], dtype=np.float32)
    feat_mean = X_all[train_idx].mean(axis=0)
    feat_std = X_all[train_idx].std(axis=0)
    feat_std[feat_std == 0] = 1.0  # 常数特征不缩放
    for r in labeled:
        r["features"] = (np.asarray(r["features"], dtype=np.float32) - feat_mean) / feat_std

    train_ds = ChartDataset(train_records)
    val_ds = ChartDataset(val_records)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = len(labeled[0]["features"])
    model = MLPBaseline(input_dim=input_dim)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"input dim: {input_dim}  params: {n_params}")
    print(model)

    loss_fn = nn.SmoothL1Loss()  # 对离群难度点更稳健（比 MSE 少被极端样本带偏）
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.out_dir, exist_ok=True)

    def evaluate(loader):
        model.eval()
        losses, preds, trues = [], [], []
        with torch.no_grad():
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                out = model(x).squeeze(-1)
                losses.append(loss_fn(out, y).item() * x.size(0))
                preds.extend(out.tolist())
                trues.extend(y.tolist())
        n = len(trues)
        mae = np.mean(np.abs(np.asarray(preds) - np.asarray(trues)))
        rmse = np.sqrt(np.mean((np.asarray(preds) - np.asarray(trues)) ** 2))
        return sum(losses) / n, mae, rmse

    # ---- 基准 A：mean predictor（永远预测训练集标签均值）----
    train_mean = float(np.mean([labeled[i]["label"] for i in train_idx]))
    val_trues = np.asarray([labeled[i]["label"] for i in val_idx], dtype=np.float32)
    mean_mae = float(np.mean(np.abs(val_trues - train_mean)))
    mean_rmse = float(np.sqrt(np.mean((val_trues - train_mean) ** 2)))
    print(f"\nbaseline A (predict train mean {train_mean:.2f}): "
          f"val_mae {mean_mae:.3f}  val_rmse {mean_rmse:.3f}")

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batch = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x).squeeze(-1)
            loss = loss_fn(out, y)
            loss.backward()      # 反向传播：对每个可学习参数计算梯度
            optimizer.step()     # 用梯度更新参数：w <- w - lr * grad
            epoch_loss += loss.item() * x.size(0)
            n_batch += 1
        train_loss = epoch_loss / len(train_idx)
        val_loss, mae, rmse = evaluate(val_loader)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_mae": mae, "val_rmse": rmse})
        print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  "
              f"val_loss {val_loss:.4f}  val_mae {mae:.3f}  val_rmse {rmse:.3f}")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "checkpoint.pt"))
    with open(os.path.join(args.out_dir, "scaler.json"), "w") as f:
        json.dump({"mean": feat_mean.tolist(), "scale": feat_std.tolist()}, f)
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "split": split_report,
            "leak": leak,
            "table": args.table,
            "smoke_test": args.smoke_test,
            "split_mode": args.split_mode,
            "mean_baseline": {"train_mean": train_mean, "val_mae": mean_mae,
                              "val_rmse": mean_rmse},
        }, f, ensure_ascii=False, indent=2)

    # 样本级预测 vs 真值
    model.eval()
    rows = []
    with torch.no_grad():
        for x, y, meta in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze(-1)
            for p, t, title, sha in zip(pred.tolist(), y.tolist(),
                                       meta["title"], meta["sha256"]):
                rows.append({"title": title, "sha256": sha,
                             "true": t, "pred": p})
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", encoding="utf-8") as f:
        f.write("title,sha256,true,pred\n")
        for row in rows:
            f.write(f"{row['title']},{row['sha256']},{row['true']:.2f},{row['pred']:.2f}\n")

    print(f"\nsaved: checkpoint.pt / metrics.json / predictions.csv -> {args.out_dir}")


if __name__ == "__main__":
    main()

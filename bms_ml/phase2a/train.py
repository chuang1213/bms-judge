"""训练与评估工具：masked reconstruction（T1）与 next-window（T2）。"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .grid_data import N_CH, N_LANES, T


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_masks_np(n: int, span_cells: int = 60, random_frac: float = 0.1,
                  seed: int = 0) -> np.ndarray:
    """生成 [n, T, L] int8：0=未遮挡，1=span，2=random cell。"""
    rng = np.random.RandomState(seed)
    m = np.zeros((n, T, N_LANES), dtype=np.int8)
    span_start = rng.randint(0, T - span_cells + 1, size=n)
    for i in range(n):
        m[i, span_start[i]:span_start[i] + span_cells, :] = 1
    rand = rng.rand(n, T, N_LANES) < random_frac
    m = np.where(rand, np.full_like(m, 2), m)
    return m


def make_masks_torch(batch: int, device, gen: torch.Generator,
                     span_cells: int = 60, random_frac: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (mask [B,T,L] bool, masks_int [B,T,L] int8)。"""
    start = torch.randint(0, T - span_cells + 1, (batch,), device=device, generator=gen)
    cells = torch.arange(T, device=device)[None, :]
    span = ((cells >= start[:, None]) & (cells < (start + span_cells)[:, None])).int()  # [B,T]
    m = span[:, :, None].expand(batch, T, N_LANES).clone()   # [B,T,L]
    rand = torch.rand((batch, T, N_LANES), device=device, generator=gen) < random_frac
    m = torch.where(rand, torch.full_like(m, 2), m)
    return m != 0, m


def _per_sample_binary_metrics(logits: torch.Tensor, targets: torch.Tensor,
                               masks_int: torch.Tensor, channel: int,
                               span_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """在 masked cells 上算 per-sample TP/FP/FN。返回 (tp, fp, fn) [B]。"""
    mask = masks_int != 0 if not span_only else masks_int == 1
    y = (targets[:, channel] > 0).float()
    p = (torch.sigmoid(logits[:, channel]) > 0.5).float()
    sel = mask
    tp = ((p == 1) & (y == 1) & sel).float().sum(dim=(1, 2))
    fp = ((p == 1) & (y == 0) & sel).float().sum(dim=(1, 2))
    fn = ((p == 0) & (y == 1) & sel).float().sum(dim=(1, 2))
    return tp, fp, fn


def _pooled_prf(tp: float, fp: float, fn: float) -> Dict[str, float]:
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"precision": float(round(prec, 4)), "recall": float(round(rec, 4)),
            "f1": float(round(f1, 4))}


def _bce_masked(logits: torch.Tensor, targets: torch.Tensor,
                masks_int: torch.Tensor, sel_value: Optional[int],
                include_hold: bool = True) -> Tuple[float, int]:
    mask = masks_int != 0 if sel_value is None else masks_int == sel_value
    loss = F.binary_cross_entropy_with_logits(logits, (targets > 0).float(),
                                              reduction="none")   # [B,2,T,L]
    mask4 = mask[:, None].expand_as(loss).clone()
    if not include_hold:
        mask4[:, 1] = False
    return float(loss[mask4].sum().item()), int(mask4.sum().item())


@torch.no_grad()
def eval_t1(model: nn.Module, grids: np.ndarray, masks_np: np.ndarray,
            device, batch_size: int = 128, span_cells: int = 60,
            include_hold: bool = True) -> Dict:
    """评估 masked reconstruction（R1）。返回 per-sample 汇总指标 + 密度分层。"""
    model = model.to(device)
    model.eval()
    n = grids.shape[0]
    tp_o = np.zeros(n); fp_o = np.zeros(n); fn_o = np.zeros(n)
    tp_h = np.zeros(n); fp_h = np.zeros(n); fn_h = np.zeros(n)
    bce_all = np.zeros(n); cnt_all = np.zeros(n)
    bce_span = np.zeros(n); cnt_span = np.zeros(n)
    dens = (grids[:, :, :, 0] > 0).sum(axis=(1, 2)).astype(np.float64)

    tens = torch.from_numpy(grids).permute(0, 3, 1, 2).to(device)
    m_t = torch.from_numpy(masks_np).to(device)
    for i in range(0, n, batch_size):
        xb = tens[i:i + batch_size]
        mb = m_t[i:i + batch_size]
        x_in = xb * (mb == 0).unsqueeze(1).float().to(device)
        if i == 0:
            leak = float((x_in * (mb != 0).unsqueeze(1).float().to(device)).abs().max().item())
            assert leak == 0.0, f"mask leak in eval input: {leak}"
        logits, _ = model.forward_mask(x_in)
        sl = slice(i, min(i + batch_size, n))
        tp_o[sl], fp_o[sl], fn_o[sl] = [v.cpu().numpy() for v in
            _per_sample_binary_metrics(logits, (xb > 0).float(), mb, 0)]
        if include_hold:
            tp_h[sl], fp_h[sl], fn_h[sl] = [v.cpu().numpy() for v in
                _per_sample_binary_metrics(logits, xb, mb, 1)]
        b_all, c_all = _bce_masked(logits, xb, mb, None, include_hold)
        b_sp, c_sp = _bce_masked(logits, xb, mb, 1, include_hold)
        bce_all[sl] = b_all; cnt_all[sl] = c_all
        bce_span[sl] = b_sp; cnt_span[sl] = c_sp

    out = {
        "onset": _pooled_prf(tp_o.sum(), fp_o.sum(), fn_o.sum()),
        "bce_all": round(float(bce_all.sum() / max(cnt_all.sum(), 1)), 5),
        "bce_span": round(float(bce_span.sum() / max(cnt_span.sum(), 1)), 5),
        "n_windows": int(n),
    }
    if include_hold:
        out["hold"] = _pooled_prf(tp_h.sum(), fp_h.sum(), fn_h.sum())
    # 按窗口密度三分位分层（只对 onset）
    q1, q2 = np.quantile(dens, [1 / 3, 2 / 3])
    for name, sel in (("low", dens <= q1), ("mid", (dens > q1) & (dens <= q2)),
                      ("high", dens > q2)):
        if sel.sum() == 0:
            out[f"stratum_{name}"] = None
            continue
        out[f"stratum_{name}"] = {
            **_pooled_prf(tp_o[sel].sum(), fp_o[sel].sum(), fn_o[sel].sum()),
            "n": int(sel.sum()),
        }
    return out


@torch.no_grad()
def eval_t2(model: nn.Module, cur_grids: np.ndarray, next_grids: np.ndarray,
            device, batch_size: int = 128) -> Dict:
    model = model.to(device)
    model.eval()
    n = cur_grids.shape[0]
    tp = np.zeros(n); fp = np.zeros(n); fn = np.zeros(n)
    cnt_pred = np.zeros(n); cnt_true = np.zeros(n)
    tens_c = torch.from_numpy(cur_grids).permute(0, 3, 1, 2).to(device)
    tens_n = torch.from_numpy(next_grids).permute(0, 3, 1, 2).to(device)
    for i in range(0, n, batch_size):
        xc = tens_c[i:i + batch_size].float()
        xn = tens_n[i:i + batch_size].float()
        logits = model(xc)
        sel = torch.ones_like(xc[:, 0], dtype=torch.bool)
        t, f, fn_ = _per_sample_binary_metrics(logits, xn, sel, 0)
        sl = slice(i, min(i + batch_size, n))
        tp[sl], fp[sl], fn[sl] = t.cpu().numpy(), f.cpu().numpy(), fn_.cpu().numpy()
        cnt_pred[sl] = (torch.sigmoid(logits[:, 0]) > 0.5).sum(dim=(1, 2)).cpu().numpy()
        cnt_true[sl] = (xn[:, 0] > 0).sum(dim=(1, 2)).cpu().numpy()
    return {
        "onset": _pooled_prf(tp.sum(), fp.sum(), fn.sum()),
        "density_mae": round(float(np.mean(np.abs(cnt_pred - cnt_true))), 4),
        "n_windows": int(n),
    }


def train_t1(model: nn.Module, tr_grids: np.ndarray, va_grids: np.ndarray,
             device, epochs: int = 20, patience: int = 6, batch_size: int = 128,
             lr: float = 1e-3, seed: int = 0,
             span_cells: int = 60, random_frac: float = 0.1,
             include_hold: bool = True) -> Tuple[nn.Module, Dict]:
    set_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dl = DataLoader(TensorDataset(torch.from_numpy(tr_grids).permute(0, 3, 1, 2)),
                    batch_size=batch_size, shuffle=True)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    best, best_state, patience_left = -float("inf"), None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        tot, cnt = 0.0, 0
        for (xb,) in dl:
            xb = xb.to(device)
            mask_bool, masks_int = make_masks_torch(xb.shape[0], device, gen,
                                                    span_cells, random_frac)
            x_in = xb * (~mask_bool).unsqueeze(1).float().to(device)
            logits, _ = model.forward_mask(x_in)
            loss_mask = mask_bool.unsqueeze(1).expand(-1, 2, T, N_LANES).clone()
            if not include_hold:
                loss_mask[:, 1] = False
            loss = F.binary_cross_entropy_with_logits(
                logits, (xb > 0).float(),
                reduction="none")[loss_mask].mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()) * xb.shape[0]; cnt += xb.shape[0]
        # val：固定 seed 的 mask 集合（取 val 的抽样子集）
        va_n = min(len(va_grids), 6000)
        rng = np.random.RandomState(seed + epoch)
        idx = np.sort(rng.choice(len(va_grids), va_n, replace=False))
        masks = make_masks_np(va_n, span_cells, random_frac, seed=1000 + epoch)
        m = eval_t1(model, va_grids[idx], masks, device, batch_size, span_cells,
                    include_hold=include_hold)
        val_f1 = m["onset"]["f1"]
        history.append({"epoch": epoch, "train_loss": round(tot / max(cnt, 1), 5),
                        "val_onset_f1": val_f1})
        if val_f1 > best + 1e-5:
            best, patience_left = val_f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_left += 1
            if patience_left >= patience:
                break
        print(f"  T1 epoch {epoch + 1}/{epochs}: train_loss {tot / max(cnt, 1):.4f} "
              f"val_onset_f1 {val_f1:.4f} (best {best:.4f})", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_onset_f1": round(float(best), 4),
                   "history": history[-30:]}


def train_t2(model: nn.Module, tr_cur: np.ndarray, tr_next: np.ndarray,
             va_cur: np.ndarray, va_next: np.ndarray,
             device, epochs: int = 12, patience: int = 4, batch_size: int = 128,
             lr: float = 1e-3, seed: int = 0) -> Tuple[nn.Module, Dict]:
    set_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.from_numpy(tr_cur).permute(0, 3, 1, 2),
                       torch.from_numpy(tr_next).permute(0, 3, 1, 2))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    best, best_state, patience_left = -float("inf"), None, 0
    history = []
    va_n = min(len(va_cur), 4000)
    rng = np.random.RandomState(seed)
    va_idx = np.sort(rng.choice(len(va_cur), va_n, replace=False))
    for epoch in range(epochs):
        model.train()
        for xc, xn in dl:
            xc, xn = xc.float().to(device), xn.float().to(device)
            logits = model(xc)
            loss = F.binary_cross_entropy_with_logits(logits, (xn > 0).float())
            opt.zero_grad(); loss.backward(); opt.step()
        m = eval_t2(model, va_cur[va_idx], va_next[va_idx], device, batch_size)
        val_f1 = m["onset"]["f1"]
        history.append({"epoch": epoch, "val_onset_f1": val_f1})
        if val_f1 > best + 1e-5:
            best, patience_left = val_f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_left += 1
            if patience_left >= patience:
                break
        print(f"  T2 epoch {epoch + 1}/{epochs}: val_onset_f1 {val_f1:.4f} "
              f"(best {best:.4f})", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_onset_f1": round(float(best), 4),
                   "history": history[-20:]}

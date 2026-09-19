"""Phase 4 provenance helper.

Every Phase 4 result JSON must carry: data config, client, split, model config,
seed, git commit. This module is the single place that assembles that header so
the anti-drift rule is enforced in code rather than in a checklist.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase4"
CLIENTS = ("beatoraja", "lr2")


def git_commit() -> str:
    """Short SHA plus a '-' suffix when the tree is dirty (result not reproducible)."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001 - provenance must never kill a run
        return "unknown"


def header(*, script: str, data_config: dict, client: str | None = None,
           split: str | None = None, model: dict | None = None,
           seed: int | None = None) -> dict:
    """Assemble the mandatory provenance block for a result JSON."""
    h = {
        "script": script,
        "git_commit": git_commit(),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "data_config": data_config,
    }
    if client is not None:
        h["client"] = client
    if split is not None:
        h["split"] = split
    if model is not None:
        h["model"] = model
    if seed is not None:
        h["seed"] = seed
    return h


def dump(path: Path, obj: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return path


def write_result(name: str, header_block: dict, payload: dict) -> Path:
    """Write `{header..., payload...}` to output/phase4/<name>.

    Flat merge (not nested) so a reader always sees seed/git/split next to the
    numbers they describe. Raises on key collision rather than silently
    overwriting provenance.
    """
    clash = set(header_block) & set(payload)
    if clash:
        raise ValueError(f"payload would overwrite provenance keys: {sorted(clash)}")
    return dump(OUT / name, {**header_block, **payload})

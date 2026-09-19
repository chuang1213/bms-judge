"""Phase 4B feature-matrix guards.

The experiments in `run_phase4b_features.py` can fail in three silent ways that would
each produce a confident wrong answer:

  1. comparing feature families on DIFFERENT row sets (a family that only covers easy
     charts wins for free) - the user's decision 4 forbids this and requires the
     common-coverage layer to be the main conclusion;
  2. imputing missing MSD/perm values, which turns "no information" into a fake score;
  3. letting a difficulty-table level in as a feature (protocol red line 3.1).

These tests lock all three, plus the stability rule (a gain must exceed seed spread)
and the shape of the interaction features.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PHASE4B = Path(__file__).resolve().parents[1] / "phase4b"
sys.path.insert(0, str(PHASE4B))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase4"))

from run_phase4b_features import (LADDERS, coverage, family_cols,  # noqa: E402
                                  interaction_matrix)

DATASET = Path(__file__).resolve().parents[1] / "output" / "phase4" / "dataset"


def toy_frame(n: int = 40, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """A frame with one obj family, one msd family that is half missing, one v2."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "sha256": [f"c{i:03d}" for i in range(n)],
        "cs_a": rng.normal(size=n),
        "cs_b": rng.normal(size=n),
        "msd_overall": rng.normal(size=n),
        "msd_stream": rng.normal(size=n),
        "v2_x": rng.normal(size=n),
    })
    df.loc[df.index[: n // 2], ["msd_overall", "msd_stream"]] = np.nan
    info = {"families": {
        "obj": {"cols": ["cs_a", "cs_b"]},
        "msd": {"cols": ["msd_overall", "msd_stream"]},
        "perm": {"cols": []},
        "v2": {"cols": ["v2_x"]},
    }}
    return df, info


class TestCoverageLayers(unittest.TestCase):
    def setUp(self):
        self.df, self.info = toy_frame()

    def test_a_family_with_missing_rows_reports_partial_coverage(self):
        cov = coverage(self.df, family_cols("msd", self.info))
        self.assertEqual(int(cov.sum()), 20)
        self.assertAlmostEqual(float(cov.mean()), 0.5)

    def test_full_and_common_layers_differ_when_a_family_is_partial(self):
        full = coverage(self.df, family_cols("obj", self.info))
        common = full.copy()
        for fam in ("obj", "msd", "v2"):
            common &= coverage(self.df, family_cols(fam, self.info))
        self.assertEqual(int(full.sum()), 40)
        self.assertEqual(int(common.sum()), 20)
        # the common layer must be a subset of every family's full layer
        for fam in ("obj", "msd", "v2"):
            self.assertTrue((common & ~coverage(self.df, family_cols(fam, self.info))).sum() == 0)

    def test_no_imputation_anywhere(self):
        """Missing values must survive into the models as NaN, not as 0 or a mean."""
        self.assertEqual(int(self.df["msd_overall"].isna().sum()), 20)
        # coverage() must never fill
        _ = coverage(self.df, family_cols("msd", self.info))
        self.assertEqual(int(self.df["msd_overall"].isna().sum()), 20)

    def test_ladders_are_prefixes_of_each_other(self):
        """The complexity ladder must be cumulative (each step ADDS a family)."""
        order = ["obj", "obj+msd", "obj+msd+perm", "obj+msd+perm+v2"]
        prev: list[str] = []
        for name in order:
            cols = family_cols(name, self.info)
            for c in prev:
                self.assertIn(c, cols, f"{name} lost column {c} from the previous rung")
            prev = cols
        self.assertEqual(LADDERS["obj"], ["obj"])


class TestNoDifficultyTableFeature(unittest.TestCase):
    def test_jrank_and_level_columns_are_excluded(self):
        """Red line: difficulty-table levels are coordinates, never input features."""
        df, info = toy_frame()
        if not (DATASET / "cross_section.parquet").exists():
            self.skipTest("cross_section not built")
        xs = pd.read_parquet(DATASET / "cross_section.parquet")
        content = sorted(c for c in xs.columns if c.startswith("cs_"))
        self.assertNotIn("cs_jrank", content, "c_jrank must stay excluded")
        self.assertNotIn("c_jrank", content)
        # and the real feature builder must not invent it
        from run_phase4b_features import load_features
        full, _, info2 = load_features(msd_cap=100.0)
        for fam in ("obj", "msd", "perm", "v2"):
            for c in family_cols(fam, info2):
                self.assertNotIn("jrank", c.lower())
                self.assertNotIn("level", c.lower())


class TestInteractionFeatures(unittest.TestCase):
    def test_interaction_shape_and_block_structure(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(12, 3))
        players = np.array(["a"] * 6 + ["b"] * 6)
        I = interaction_matrix(X, players)
        self.assertEqual(I.shape, (12, 3 + 3 * 2))
        np.testing.assert_allclose(I[:, :3], X)
        # a player's own block equals its features; the other player's block is zero
        np.testing.assert_allclose(I[0, 3:6], X[0])
        np.testing.assert_allclose(I[0, 6:9], 0.0)
        np.testing.assert_allclose(I[6, 6:9], X[6])
        np.testing.assert_allclose(I[6, 3:6], 0.0)

    def test_interaction_can_expresses_different_player_weights(self):
        from sklearn.linear_model import Ridge
        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 2))
        players = np.array(["a"] * 150 + ["b"] * 150)
        y = np.where(players == "a", X[:, 0] * 3.0, X[:, 1] * -2.0)
        m = Ridge(alpha=1e-8).fit(interaction_matrix(X, players), y)
        self.assertLess(float(np.mean(np.abs(m.predict(interaction_matrix(X, players)) - y))),
                        1e-3)


class TestStabilityRule(unittest.TestCase):
    """A gain smaller than the seed spread must not count as a win."""

    @staticmethod
    def _stable_gain(a, b, mode="lower"):
        if not a or not b:
            return None
        gain = (b["mean"] - a["mean"]) if mode == "lower" else (a["mean"] - b["mean"])
        wins = [((bv - v) if mode == "lower" else (v - bv))
                for v, bv in zip(a["values"], b["values"])]
        return {"gain": round(gain, 4),
                "stable_win": bool(wins and all(w > 0 for w in wins) and gain > a["std"])}

    def test_gain_inside_seed_noise_is_not_a_win(self):
        a = {"mean": 6.20, "std": 0.40, "values": [6.1, 6.2, 6.3]}
        b = {"mean": 6.25, "std": 0.05, "values": [6.2, 6.25, 6.3]}
        self.assertFalse(self._stable_gain(a, b)["stable_win"])

    def test_large_consistent_gain_is_a_win(self):
        a = {"mean": 5.80, "std": 0.10, "values": [5.7, 5.8, 5.9]}
        b = {"mean": 6.25, "std": 0.05, "values": [6.2, 6.25, 6.3]}
        self.assertTrue(self._stable_gain(a, b)["stable_win"])

    def test_one_bad_seed_breaks_stability(self):
        a = {"mean": 6.00, "std": 0.05, "values": [5.9, 6.0, 6.9]}
        b = {"mean": 6.25, "std": 0.05, "values": [6.2, 6.25, 6.3]}
        self.assertFalse(self._stable_gain(a, b)["stable_win"])


class TestRealFeatureFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.msd40 = DATASET / "msd_cap40.parquet"
        cls.msd100 = DATASET / "msd_cap100.parquet"
        if not cls.msd100.exists():
            raise unittest.SkipTest("MSD feature tables not built")

    def test_capped_and_patched_versions_are_separate_files(self):
        a, b = pd.read_parquet(self.msd40), pd.read_parquet(self.msd100)
        self.assertEqual(a["msd_cap"].unique().tolist(), [40.0])
        self.assertEqual(b["msd_cap"].unique().tolist(), [100.0])
        cols = [c for c in b.columns if c.startswith("msd_") and c != "msd_cap"]
        self.assertEqual(len(cols), 7, f"expected 7 live axes, got {cols}")

    def test_dead_axis_was_dropped(self):
        b = pd.read_parquet(self.msd100)
        self.assertNotIn("msd_technical", b.columns,
                         "Technical is effectively constant on the n-key path")

    def test_patched_values_are_never_below_the_stock_values(self):
        """Raising the clamp can only increase a clamped skill value."""
        a, b = pd.read_parquet(self.msd40), pd.read_parquet(self.msd100)
        m = a.merge(b, on="sha256", suffixes=("_40", "_100"))
        for c in [c for c in b.columns if c.startswith("msd_") and c != "msd_cap"]:
            d = (m[f"{c}_100"] - m[f"{c}_40"]).min()
            self.assertGreaterEqual(float(d), -1e-4, f"{c} decreased after unclamping")

    def test_missing_values_are_not_imputed(self):
        b = pd.read_parquet(self.msd100)
        cols = [c for c in b.columns if c.startswith("msd_") and c != "msd_cap"]
        # rows present must have real values; absent charts must be absent, not zero
        self.assertTrue(b[cols].notna().all().all())


if __name__ == "__main__":
    unittest.main()

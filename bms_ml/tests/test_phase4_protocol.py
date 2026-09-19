"""Phase 4 protocol guards.

These lock the two things a Phase 4 mistake would silently corrupt: the split
definitions (a leak would inflate every later number) and the baseline
decomposition (a baseline fitted on test rows would make the M3 gate meaningless).

Data-free: synthetic frames only, so this runs anywhere.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase4"))

from evaluate_matrix import BASELINES, GATE_BASELINE, load_frame  # noqa: E402
from metrics import mae  # noqa: E402
from models import BaselineModel  # noqa: E402
from splits import split_mask  # noqa: E402


def fit_model(tr: pd.DataFrame, model: str, target: str) -> BaselineModel:
    """Backwards-compatible helper: the decompose-then-predict baselines.

    `fit_model`/`predict` used to be module-level functions in evaluate_matrix;
    the model now lives in models.BaselineModel, so these two shims keep the
    protocol tests expressing the same invariant.
    """
    return BaselineModel(model).fit(tr, target)


def predict(te: pd.DataFrame, model: str, fit: BaselineModel, target: str) -> np.ndarray:
    return fit.predict(te)


def synthetic(n_players: int = 6, n_charts: int = 40, seed: int = 0) -> pd.DataFrame:
    """A sparse (player x chart) frame with a known additive structure."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        for c in rng.choice(n_charts, size=12, replace=False):
            rows.append({"player": f"p{p}", "sha256": f"c{c:03d}",
                         "acc": float(np.clip(50 + 5 * p - 0.7 * c + rng.normal(0, 2), 0, 100))})
    return pd.DataFrame(rows).reset_index(drop=True)


class TestSplitsDoNotLeak(unittest.TestCase):
    def setUp(self):
        self.df = synthetic()

    def _sides(self, kind: str):
        test = split_mask(self.df, kind, 0, 0.25)
        return self.df[~test], self.df[test], test

    def test_splits_partition_the_rows(self):
        for kind in ("random_interaction", "loo_1_charts_per_player", "cold_player", "cold_chart"):
            tr, te, test = self._sides(kind)
            self.assertEqual(len(tr) + len(te), len(self.df), kind)
            self.assertFalse((test.sum() == 0) or (test.sum() == len(self.df)), kind)

    def test_no_player_chart_cell_appears_on_both_sides(self):
        for kind in ("random_interaction", "loo_1_charts_per_player", "cold_player", "cold_chart"):
            tr, te, _ = self._sides(kind)
            trk = set(zip(tr["player"], tr["sha256"]))
            tek = set(zip(te["player"], te["sha256"]))
            self.assertEqual(trk & tek, set(), f"{kind} reuses a cell")

    def test_cold_player_holds_out_whole_players(self):
        tr, te, _ = self._sides("cold_player")
        self.assertEqual(set(te["player"]) & set(tr["player"]), set())
        # the warm players must still be there; a cold split is not "drop everything"
        self.assertTrue(len(tr) > 0 and set(tr["player"]))

    def test_cold_chart_holds_out_whole_charts(self):
        tr, te, _ = self._sides("cold_chart")
        self.assertEqual(set(te["sha256"]) & set(tr["sha256"]), set())

    def test_warm_splits_keep_every_player_and_chart_in_train(self):
        for kind in ("random_interaction", "loo_1_charts_per_player"):
            tr, te, _ = self._sides(kind)
            self.assertEqual(set(te["player"]) - set(tr["player"]), set(), kind)
            # warm splits must not accidentally become cold-chart splits
            self.assertGreater(len(set(tr["sha256"]) & set(te["sha256"])), 0, kind)

    def test_loo_holds_out_exactly_one_row_per_player(self):
        _, te, _ = self._sides("loo_1_charts_per_player")
        self.assertEqual(set(te["player"].value_counts().unique()), {1})

    def test_same_seed_reproduces_the_same_split(self):
        a = split_mask(self.df, "random_interaction", 1, 0.25)
        b = split_mask(self.df, "random_interaction", 1, 0.25)
        c = split_mask(self.df, "random_interaction", 2, 0.25)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c), "seed has no effect")


class TestBaselinesAreTrainOnly(unittest.TestCase):
    def setUp(self):
        self.df = synthetic()
        test = split_mask(self.df, "random_interaction", 0, 0.25)
        self.tr, self.te = self.df[~test].copy(), self.df[test].copy()

    def test_predictions_do_not_depend_on_test_targets(self):
        """Tampering with test labels must not move the predictions at all."""
        for m in BASELINES:
            model = fit_model(self.tr, m, "acc")
            p1 = model.predict(self.te)
            poisoned = self.te.copy()
            poisoned["acc"] = poisoned["acc"] + 1000.0
            p2 = model.predict(poisoned)
            np.testing.assert_allclose(p1, p2, err_msg=f"{m} leaks test targets")

    def test_decomposition_is_additive_and_centered(self):
        fit = fit_model(self.tr, "player_chart_bias", "acc")
        # Effects are centered by construction: each is a group mean minus the
        # global mean, so the ROW-weighted mean over training rows is 0 (a plain
        # unweighted mean over players is not, because players have unequal n).
        w_p = self.tr.groupby("player").size()
        w_c = self.tr.groupby("sha256").size()
        self.assertAlmostEqual(float((fit.player_eff * w_p).sum() / len(self.tr)), 0.0, places=9)
        self.assertAlmostEqual(float((fit.chart_eff * w_c).sum() / len(self.tr)), 0.0, places=9)
        p = fit.predict(self.te)
        manual = (fit.mu
                  + self.te["player"].map(fit.player_eff).fillna(0.0).to_numpy()
                  + self.te["sha256"].map(fit.chart_eff).fillna(0.0).to_numpy())
        np.testing.assert_allclose(p, manual)

    def test_unseen_player_and_chart_get_zero_effect(self):
        fit = fit_model(self.tr, "player_chart_bias", "acc")
        te = self.te.copy()
        te.loc[te.index[0], "player"] = "never_seen_player"
        te.loc[te.index[1], "sha256"] = "never_seen_chart"
        p = fit.predict(te)
        self.assertFalse(np.isnan(p).any())
        self.assertAlmostEqual(float(p[0]), float(fit.mu
                                                  + fit.chart_eff.get(te["sha256"].iloc[0], 0.0)))
        self.assertAlmostEqual(float(p[1]), float(fit.mu
                                                  + fit.player_eff.get(te["player"].iloc[1], 0.0)))

    def test_gate_baseline_beats_global_mean_on_fitting_data(self):
        """Sanity that the decomposition is actually being fitted (not a constant)."""
        fit = fit_model(self.tr, GATE_BASELINE, "acc")
        glob = fit_model(self.tr, "global_mean", "acc")
        self.assertLess(mae(self.tr["acc"].to_numpy(), fit.predict(self.tr)),
                        mae(self.tr["acc"].to_numpy(), glob.predict(self.tr)))

    def test_cold_player_falls_back_to_global_mean(self):
        test = split_mask(self.df, "cold_player", 0, 0.34)
        tr, te = self.df[~test], self.df[test]
        fit = fit_model(tr, "player_chart_bias", "acc")
        p = fit.predict(te)
        cold = ~te["player"].isin(tr["player"]).to_numpy()
        self.assertTrue(cold.any(), "cold_player split produced no cold player")
        # A cold player has no player effect, so only the chart effect may move the
        # prediction away from mu.
        expected = (fit.mu
                    + te["sha256"].map(fit.chart_eff).fillna(0.0).to_numpy())[cold]
        np.testing.assert_allclose(p[cold], expected)


class TestRealCrossSection(unittest.TestCase):
    """Integration guards against the built artifact.

    Skipped when `cross_section.parquet` has not been built, so the unit suite
    still runs on a clean checkout. When it IS present, these are the checks that
    would catch a client being pooled, a target escaping [0,100], or a split
    leaking on real data.
    """

    PARQUET = (Path(__file__).resolve().parents[1] / "output" / "phase4" / "dataset"
               / "cross_section.parquet")
    FILTERED = None

    @classmethod
    def setUpClass(cls):
        if not cls.PARQUET.exists():
            raise unittest.SkipTest(f"{cls.PARQUET.name} not built yet")
        cls.raw = pd.read_parquet(cls.PARQUET)
        cls.filtered, _ = load_frame(cls.PARQUET, "acc", 120)

    def test_clients_are_never_pooled_into_one_frame(self):
        self.assertEqual(set(self.raw["client"]), {"beatoraja", "lr2"})
        # every (player, sha) pair belongs to exactly one client
        pair_clients = self.raw.groupby(["player", "sha256"])["client"].nunique()
        self.assertEqual(int(pair_clients.max()), 1)

    def test_lamp_scales_are_client_specific(self):
        bj = self.raw[self.raw["client"] == "beatoraja"]["lamp"]
        lr = self.raw[self.raw["client"] == "lr2"]["lamp"]
        self.assertLessEqual(int(bj.max()), 10)
        self.assertLessEqual(int(lr.max()), 5)   # LR2 archive carries no FC/PERFECT rows

    def test_target_is_a_percentage_and_denominator_is_sane(self):
        acc = self.raw["acc"].dropna()
        self.assertGreaterEqual(float(acc.min()), 0.0)
        self.assertLessEqual(float(acc.max()), 100.0)
        self.assertGreater(int((self.raw["notes"].fillna(0) > 0).sum()), 0)

    def test_sha_keys_are_unique_per_player_and_client(self):
        d = self.raw.duplicated(["client", "player", "sha256"]).sum()
        self.assertEqual(int(d), 0)

    def test_real_splits_do_not_leak(self):
        for client in ("beatoraja", "lr2"):
            df = self.filtered[self.filtered["client"] == client].reset_index(drop=True)
            if df["player"].nunique() < 2:
                continue        # cold_player is undefined for a single-player matrix
            for kind in ("random_interaction", "loo_1_charts_per_player", "cold_chart"):
                test = split_mask(df, kind, 0, 0.2)
                tr, te = df[~test], df[test]
                trk = set(zip(tr["player"], tr["sha256"]))
                tek = set(zip(te["player"], te["sha256"]))
                self.assertEqual(trk & tek, set(), f"{client}/{kind} reuses a cell")
                if kind == "cold_chart":
                    self.assertEqual(set(te["sha256"]) & set(tr["sha256"]), set(),
                                     f"{client}/{kind} leaks charts")
            test = split_mask(df, "cold_player", 0, 0.2)
            tr, te = df[~test], df[test]
            self.assertEqual(set(te["player"]) & set(tr["player"]), set(),
                             f"{client}/cold_player leaks players")


if __name__ == "__main__":
    unittest.main()

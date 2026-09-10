"""Phase 3 feature-registry / evaluation-primitive regression tests.

Guard for a real failure that happened (found 2026-09-07): `c_jrank` was added to
every experiment script's local feature list but never to
`chart_repr.OBJECTIVE_STAT_COLS`, while `chart_encoder_registry()` already
declared 27 dims. The registry — the module PROTOCOL.md §3 designates as the
single source of truth — had drifted out of sync with the code that actually
trained the models, and nothing detected it.

These tests are data-free (samples.parquet is git-ignored), so they run anywhere.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bms_ml.phase3 import chart_repr
from bms_ml.phase3.common import HGBModel, Imputer, centered_r2, hgb_fit_predict, mae, r2


class TestFeatureRegistry(unittest.TestCase):
    def test_objective_stat_dims_match_registry(self):
        """The declared encoder dim must equal the actual list length."""
        desc, dim = chart_repr.chart_encoder_registry()["objective_stats"]
        self.assertEqual(dim, len(chart_repr.OBJECTIVE_STAT_COLS),
                         f"registry declares {dim} dims ({desc}) but the list has "
                         f"{len(chart_repr.OBJECTIVE_STAT_COLS)}")

    def test_jrank_present(self):
        # #RANK judge-window tier; without it tight-rank charts are systematically
        # over-predicted by 5-11pp acc (EXPERIMENT_LOG 2026-09-05).
        self.assertIn("c_jrank", chart_repr.OBJECTIVE_STAT_COLS)

    def test_no_difficulty_table_features(self):
        """PROTOCOL.md §1: table level is a fence/coordinate, never a feature."""
        banned = {"level", "table", "c_level", "c_table", "level_norm",
                  "h_level_acc", "table_satellite", "table_stella", "table_insane"}
        allf = set(chart_repr.OBJECTIVE_STAT_COLS) | set(chart_repr.HISTORY_FEATURES)
        self.assertEqual(allf & banned, set())

    def test_few_shot_schema_is_a_subset(self):
        few = chart_repr.HISTORY_FEW_FEATURES
        self.assertTrue(set(few) <= set(chart_repr.HISTORY_FEATURES))
        # the whole point of the split: recency terms are excluded
        for c in chart_repr.HISTORY_TIME_FEATURES:
            self.assertNotIn(c, few)
        self.assertEqual(len(few) + len(chart_repr.HISTORY_TIME_FEATURES),
                         len(chart_repr.HISTORY_FEATURES))

    def test_feature_manifest_flags_table_use(self):
        m = chart_repr.feature_manifest(chart_repr.OBJECTIVE_STAT_COLS)
        self.assertFalse(m["uses_difficulty_table_features"])
        self.assertEqual(m["feature_lists"][0], list(chart_repr.OBJECTIVE_STAT_COLS))


class TestEvalPrimitives(unittest.TestCase):
    def test_mae_and_r2(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(mae(y, y), 0.0)
        self.assertAlmostEqual(mae(y, y + 1), 1.0)
        self.assertAlmostEqual(r2(y, y), 1.0)
        self.assertAlmostEqual(r2(y, np.full(4, y.mean())), 0.0)

    def test_centered_r2_ignores_player_strength(self):
        """A model that only knows each player's mean must score ~0, not ~1."""
        d = pd.DataFrame({"player": ["a"] * 3 + ["b"] * 3,
                          "acc": [80.0, 82.0, 84.0, 50.0, 52.0, 54.0]})
        player_mean = d.groupby("player")["acc"].transform("mean").values
        self.assertAlmostEqual(centered_r2(d, player_mean, "acc"), 0.0, places=9)
        self.assertAlmostEqual(centered_r2(d, d["acc"].values, "acc"), 1.0, places=9)

    def test_imputer_fits_on_train_only(self):
        tr = pd.DataFrame({"x": [1.0, 3.0, np.nan]})
        te = pd.DataFrame({"x": [np.nan, 5.0]})
        imp = Imputer().fit(tr)
        np.testing.assert_allclose(imp.transform(te), [[2.0], [5.0]])

    def test_imputer_all_nan_column_becomes_zero(self):
        # happens for real: a fully-masked time feature in time_ablation's
        # H_masked_100 variant must collapse to 0, not propagate NaN
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            imp = Imputer().fit(pd.DataFrame({"x": [np.nan, np.nan]}))
            out = imp.transform(pd.DataFrame({"x": [np.nan]}))
        np.testing.assert_allclose(out, [[0.0]])


class TestHGBModelHoisting(unittest.TestCase):
    """Guard for the 2026-09-11 speedup refactor.

    The few-shot k-loop varies ONLY the evaluation frame, so transfer_eval /
    c0_state_hgb used to refit the identical HGB once per k (~8x redundant). They now
    hoist the fit through `common.HGBModel`. That is only legitimate if it reproduces
    the one-shot path exactly — otherwise every reported few-shot number silently
    shifts. Data-free so it runs anywhere.
    """

    def test_hgbmodel_matches_hgb_fit_predict(self):
        rng = np.random.RandomState(0)
        tr = pd.DataFrame(rng.normal(size=(300, 5)), columns=list("abcde"))
        tr.iloc[::7, 0] = np.nan                    # exercise the imputer
        tr["y"] = rng.normal(size=300)
        te = pd.DataFrame(rng.normal(size=(60, 5)), columns=list("abcde"))
        te.iloc[::5, 1] = np.nan
        np.testing.assert_array_equal(
            hgb_fit_predict(tr, te, list("abcde"), tr["y"].values),
            HGBModel(tr, list("abcde"), tr["y"].values).predict(te))

    def test_hgbmodel_is_reusable_across_frames(self):
        """The k-loop contract: one fit, many evaluation frames, same predictions."""
        rng = np.random.RandomState(1)
        tr = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
        tr["y"] = rng.normal(size=200)
        m = HGBModel(tr, list("abcd"), tr["y"].values)
        for _ in range(3):
            te = pd.DataFrame(rng.normal(size=(20, 4)), columns=list("abcd"))
            np.testing.assert_array_equal(
                m.predict(te),
                hgb_fit_predict(tr, te, list("abcd"), tr["y"].values))


if __name__ == "__main__":
    unittest.main()

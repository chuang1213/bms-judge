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

    def test_declared_dims_match_lists(self):
        """Every list-backed encoder must declare its real list length.

        Same failure class as the `c_jrank` drift (2026-09-07): the registry said 27
        dims while the list had 26. perm_space (2026-09-11) joins the guard.
        """
        reg = chart_repr.chart_encoder_registry()
        pairs = {
            "objective_stats": chart_repr.OBJECTIVE_STAT_COLS,
            "objective_stats_v2": (chart_repr.OBJECTIVE_STAT_COLS
                                   + chart_repr.OBJECTIVE_V2_COLS),
            "perm_space": chart_repr.OBJECTIVE_PERM_COLS,
        }
        for name, cols in pairs.items():
            self.assertIn(name, reg)
            self.assertEqual(reg[name][1], len(cols), f"{name} dim mismatch")

    def test_response_profile_lists_align(self):
        """history_response.py must emit exactly the registered columns.

        Guards the 2026-09-11 personal-response-profile encoder: the writer and the
        registry drifted apart once already for a different list (c_jrank), and a
        silent drift here would produce an all-NaN feature block that HGB imputes
        away without any error.
        """
        axes = chart_repr.RESPONSE_AXES
        want = ([f"h_resp_{k}" for k in axes] + [f"h_slope_{k}" for k in axes]
                + ["h_resp_mean", "h_resp_std"])
        self.assertEqual(chart_repr.HISTORY_RESPONSE_COLS, want)
        self.assertEqual(len(chart_repr.HISTORY_RESPONSE_COLS), 2 * len(axes) + 2)
        # the lamp block must be the same shape under its own prefix
        lamp = chart_repr.HISTORY_RESPONSE_LAMP_COLS
        self.assertEqual(lamp, [c.replace("h_", "l_", 1)
                                for c in chart_repr.HISTORY_RESPONSE_COLS])
        self.assertEqual(len(lamp), 2 * len(axes) + 2)
        allc = set(chart_repr.OBJECTIVE_STAT_COLS) | set(chart_repr.OBJECTIVE_V2_COLS)
        for k, col in axes.items():
            self.assertIn(col, allc, f"response axis {k} -> {col} is not objective")

    def test_no_difficulty_table_features(self):
        """PROTOCOL.md §1: table level is a fence/coordinate, never a feature."""
        banned = {"level", "table", "c_level", "c_table", "level_norm",
                  "h_level_acc", "table_satellite", "table_stella", "table_insane"}
        allf = (set(chart_repr.OBJECTIVE_STAT_COLS)
                | set(chart_repr.HISTORY_FEATURES)
                | set(chart_repr.OBJECTIVE_V2_COLS)
                | set(chart_repr.OBJECTIVE_PERM_COLS)
                | set(chart_repr.HISTORY_RESPONSE_COLS))
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


class TestLR2Labels(unittest.TestCase):
    """LR2 archives enter the project as player state (2026-09-11, lr2_reader.py).

    The label formula is the one thing that must be right before the data is usable:
    LR2 EX score is perfect*2 + great over a max of notes*2. Verified on the real
    archive against its own integer `rate` column.
    """

    def test_acc_formula(self):
        from bms_ml.phase3.lr2_reader import compute_acc
        # 1131 PG + 374 GR over 1530 notes = 2636/3060 = 86.14% (the archive stores 86)
        self.assertAlmostEqual(float(compute_acc([1131], [374], [1530])[0]),
                               86.1438, places=3)
        self.assertAlmostEqual(float(compute_acc([10], [0], [10])[0]), 100.0)
        self.assertAlmostEqual(float(compute_acc([0], [0], [10])[0]), 0.0)
        self.assertTrue(np.isnan(compute_acc([1], [1], [0])[0]))

    def test_clear_codes_are_gauge_types(self):
        from bms_ml.phase3.lr2_reader import LR2_CLEAR
        # the archive only carries 0..5; FC/PERFECT are declared but absent, which is
        # exactly why mixing an LR2 lamp with a beatoraja lamp is not allowed
        self.assertEqual(LR2_CLEAR[1], "FAILED")
        self.assertEqual(LR2_CLEAR[5], "EXHARD")
        self.assertEqual(LR2_CLEAR[7], "PERFECT")

    def test_lr2_to_beatoraja_lamp_mapping(self):
        """The gauge ladder is the same, beatoraja just inserts two ASSIST levels at 2,3.

        Both enums are transcribed from beatoraja-master (see lr2_reader.py). This
        mapping exists so a cross-client comparison is explicit - it does NOT make the
        two clients interchangeable.
        """
        from bms_ml.phase3.lr2_reader import (BEATORAJA_CLEAR, lr2_clear_to_beatoraja)
        self.assertEqual(BEATORAJA_CLEAR[0], "NO_PLAY")
        self.assertEqual(BEATORAJA_CLEAR[1], "FAILED")
        self.assertEqual(BEATORAJA_CLEAR[4], "EASY")
        self.assertEqual(BEATORAJA_CLEAR[6], "HARD")
        self.assertEqual(BEATORAJA_CLEAR[8], "FC")
        # gauge ladder shifts by exactly +2
        for lr2_v, want in [(0, 0), (1, 1), (2, 4), (3, 5), (4, 6), (5, 7),
                            (6, 8), (7, 9), (8, 10), (9, 2), (10, 3)]:
            self.assertEqual(lr2_clear_to_beatoraja(lr2_v), want, f"lr2 clear {lr2_v}")
        # and the shift really does land on the same gauge name
        self.assertEqual(BEATORAJA_CLEAR[lr2_clear_to_beatoraja(2)], "EASY")
        self.assertEqual(BEATORAJA_CLEAR[lr2_clear_to_beatoraja(5)], "EXHARD")
        with self.assertRaises(ValueError):
            lr2_clear_to_beatoraja(99)


class TestClientGuard(unittest.TestCase):
    """LR2 and beatoraja must not be pooled (user decision 2026-09-11).

    Measured on 4,576 charts present in both of vsoflan's archives: the acc gap has
    sd 10.16pp and exceeds 5pp on 29.4% of charts, which is larger than the model's own
    acc MAE. So mixing them would inject noise comparable to the signal, and the builder
    has to refuse rather than degrade quietly.
    """

    def test_refuses_non_beatoraja(self):
        from bms_ml.phase3.data import assert_single_client
        with self.assertRaises(SystemExit):
            assert_single_client({"vsoflan_lr2": {"client": "lr2"}})

    def test_accepts_all_beatoraja(self):
        from bms_ml.phase3.data import assert_single_client
        assert_single_client({"a": {"client": "beatoraja"}, "b": {}})   # default = beatoraja

    def test_default_is_beatoraja(self):
        """A roster entry written before the client field existed must still build."""
        from bms_ml.phase3.data import assert_single_client
        assert_single_client({"legacy_player": {"dir": "PlayerData/beatoraja/x/player1"}})


if __name__ == "__main__":
    unittest.main()

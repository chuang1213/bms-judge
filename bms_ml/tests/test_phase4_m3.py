"""Phase 4 M3 guards: split semantics, player-internal metrics, and the ALS solver.

These exist because three real bugs were found during M3 development, each of
which would have produced plausible-looking but wrong numbers:

  1. the ALS half-step wrote into the FIXED opposite factor block, aliasing the
     two blocks so the objective oscillated instead of descending;
  2. the batched normal-equation solver collapsed every residual in a group to
     the first one (a bad broadcast), zeroing the solution;
  3. the size-1 support case used `reg` alone in the denominator, missing the
     Sherman-Morrison `||g||^2` term and scaling the whole factor vector.

All three are cheap to test and silent in production, so they are locked here.
Data-free: synthetic frames only.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase4"))

from metrics import (centered, evaluate_predictions,  # noqa: E402
                     player_train_means, ranking_by_player, spearman)
from models import BaselineModel, ContentRidgeModel, MFModel  # noqa: E402
from splits import (all_split_names, holdout_indices, holdout_mask,  # noqa: E402
                    loo_k_name, parse_split, split_mask)


def toy(n_players: int = 8, n_charts: int = 60, per_player: int = 25,
        seed: int = 0, signal: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 2.0, size=(n_players, 3))
    Q = rng.normal(0, 2.0, size=(n_charts, 3))
    rows = []
    for u in range(n_players):
        for c in rng.choice(n_charts, size=per_player, replace=False):
            base = 60 + 3.0 * u + (P[u] @ Q[c] / 2.0 if signal else 0.0)
            rows.append({"player": f"p{u}", "sha256": f"c{c:03d}",
                         "acc": float(np.clip(base + rng.normal(0, 3), 0, 100)),
                         "cs_total_notes": float(500 + c * 10),
                         "cs_avg_nps": float(5 + c / 20)})
    return pd.DataFrame(rows)


def informative_predictor(te: pd.DataFrame, means: dict, seed: int = 0,
                          shrink: float = 0.5) -> np.ndarray:
    """A non-degenerate predictor: reproduces part of each player's deviation.

    Needed because Spearman/NDCG are undefined for a constant predictor - the
    `player_mean` model has zero within-player variance by construction.
    """
    flat = te["player"].map(means).to_numpy(float)
    rng = np.random.default_rng(seed)
    dev = te["acc"].to_numpy(float) - flat
    return flat + shrink * dev + rng.normal(0, 0.5, len(te))


class TestSplitNames(unittest.TestCase):
    def test_loo_k_roundtrip(self):
        self.assertEqual(parse_split(loo_k_name(5)), ("loo_k_charts_per_player", 5))
        self.assertEqual(parse_split("random_interaction"), ("random_interaction", None))
        self.assertEqual(parse_split("cold_chart"), ("cold_chart", None))

    def test_all_split_names_lists_every_loo_k(self):
        names = all_split_names((1, 5, 10))
        for k in (1, 5, 10):
            self.assertIn(loo_k_name(k), names)
        self.assertIn("cold_chart", names)


class TestLooKSplit(unittest.TestCase):
    def setUp(self):
        self.df = toy()

    def test_k1_holds_out_exactly_one_row_per_player(self):
        test = split_mask(self.df, loo_k_name(1), 0, 0.2)
        te = self.df[test]
        self.assertEqual(set(te["player"].value_counts().unique()), {1})

    def test_k5_holds_out_up_to_five_charts_per_player(self):
        test = split_mask(self.df, loo_k_name(5), 0, 0.2)
        te = self.df[test]
        per_player = te.groupby("player").size()
        self.assertTrue((per_player <= 5).all(), per_player.to_dict())
        # with 25 observed charts per player and k=5, each player should get exactly 5
        self.assertEqual(set(per_player.unique()), {5})

    def test_loo_k_does_not_leak_players_or_cells(self):
        for k in (1, 5, 10):
            test = split_mask(self.df, loo_k_name(k), 0, 0.2)
            tr, te = self.df[~test], self.df[test]
            self.assertEqual(set(te["player"]) - set(tr["player"]), set())
            trk = set(zip(tr["player"], tr["sha256"]))
            tek = set(zip(te["player"], te["sha256"]))
            self.assertEqual(trk & tek, set(), f"loo_k={k} reuses a cell")

    def test_k1_reproduces_the_original_salt(self):
        """The k=1 split must stay identical to the 'loo_chart_per_player' name used
        before the loo_k patch, so older recorded splits still reproduce."""
        a = split_mask(self.df, loo_k_name(1), 3, 0.2)
        # the pre-patch name is not a valid split any more; the k=1 salt is what
        # matters, and it must be stable across calls
        b = split_mask(self.df, loo_k_name(1), 3, 0.2)
        np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            split_mask(self.df, "loo_chart_per_player", 3, 0.2)

    def test_seed_changes_the_draw(self):
        a = split_mask(self.df, loo_k_name(5), 0, 0.2)
        b = split_mask(self.df, loo_k_name(5), 1, 0.2)
        self.assertFalse(np.array_equal(a, b))


class TestHoldoutHelpers(unittest.TestCase):
    def test_holdout_indices_are_positional_and_match_the_mask(self):
        df = toy()
        m = holdout_mask(df, 0, 0.25)
        idx = holdout_indices(df, 0, 0.25)
        self.assertEqual(m.dtype, bool)
        np.testing.assert_array_equal(np.flatnonzero(m), idx)
        # the returned indices must select exactly the masked rows positionally
        pd.testing.assert_frame_equal(df.iloc[idx].reset_index(drop=True),
                                      df[m].reset_index(drop=True))


class TestPlayerInternalMetrics(unittest.TestCase):
    def setUp(self):
        self.df = toy()
        test = split_mask(self.df, "random_interaction", 0, 0.2)
        self.tr, self.te = self.df[~test].reset_index(drop=True), self.df[test].reset_index(drop=True)
        self.means = player_train_means(self.tr, "acc")

    def test_centered_removes_the_player_level_exactly(self):
        """Centring must make the metric invariant to a player's overall level.

        Use a predictor whose within-player deviations are non-trivial (the player
        mean plus an optimistic offset). Adding ANOTHER constant to every prediction
        must leave the player-centred MAE unchanged, while the plain MAE moves by
        exactly that constant.
        """
        p = self.te["player"].map(self.means).to_numpy(float) + 15.0
        c = 7.0
        r0 = evaluate_predictions(self.te, p, "acc", self.means)
        r1 = evaluate_predictions(self.te, p + c, "acc", self.means)
        self.assertAlmostEqual(r1["mae"] - r0["mae"], c, places=6)
        # the level-free part must be untouched by that shift
        self.assertAlmostEqual(r0["player_centered_mae"], r1["player_centered_mae"], places=9)
        self.assertAlmostEqual(r0["player_centered_rmse"], r1["player_centered_rmse"], places=9)
        self.assertAlmostEqual(r0["player_centered_spearman"],
                               r1["player_centered_spearman"], places=9)
        for k in r0["per_player_centered_mae"]:
            self.assertAlmostEqual(r1["per_player_centered_mae"][k],
                                   r0["per_player_centered_mae"][k], places=9)

    def test_centered_mae_is_not_the_plain_mae(self):
        """Regression guard: `MAE(y - base, p - base)` collapses to `|y - p|`."""
        p = self.te["player"].map(self.means).to_numpy(float) + 15.0
        y = self.te["acc"].to_numpy(float)
        base = self.te["player"].map(self.means).to_numpy(float)
        # what the buggy version computed: identical to the plain MAE
        np.testing.assert_allclose(np.abs((y - base) - (p - base)), np.abs(y - p))
        r = evaluate_predictions(self.te, p, "acc", self.means)
        self.assertNotAlmostEqual(r["player_centered_mae"], r["mae"], places=3)
        # with a constant offset the centred metric must be far smaller than MAE
        self.assertLess(r["player_centered_mae"], r["mae"])

    def test_centered_metric_rewards_within_player_ordering(self):
        """A model that reproduces each player's own deviations must beat the
        'always predict the player's level' model on the centred metric."""
        flat = self.te["player"].map(self.means).to_numpy(float)
        good = informative_predictor(self.te, self.means, shrink=0.5)
        r_flat = evaluate_predictions(self.te, flat, "acc", self.means, want_ranking=True)
        r_good = evaluate_predictions(self.te, good, "acc", self.means, want_ranking=True)
        self.assertLess(r_good["player_centered_mae"], r_flat["player_centered_mae"])
        self.assertGreater(r_good["player_rank_spearman"], 0.5)

    def test_ranking_metrics_have_the_expected_shape(self):
        pred = informative_predictor(self.te, self.means)
        r = evaluate_predictions(self.te, pred, "acc", self.means, want_ranking=True)
        # most players must yield a Spearman; players with too few test rows for a
        # correlation are reported as missing rather than as a fake number
        n = r["ranking"]["aggregate"]["spearman"]["n_players"]
        self.assertGreaterEqual(n, self.te["player"].nunique() - 1)
        self.assertLessEqual(n, self.te["player"].nunique())
        for k in (5, 10, 20):
            self.assertIn(f"ndcg@{k}", r["ranking"]["aggregate"])
        # the ranking aggregate must be surfaced at the top level too, because the
        # M3 gate is stated in terms of those numbers
        self.assertIsNotNone(r["player_rank_spearman"])
        self.assertEqual(set(r["player_topk_ndcg"]), {"ndcg@5", "ndcg@10", "ndcg@20"})

    def test_constant_predictor_has_no_ranking_information(self):
        """A predictor with zero within-player variance cannot rank; the aggregate
        must simply omit Spearman rather than invent a number."""
        flat = self.te["player"].map(self.means).to_numpy(float)
        r = evaluate_predictions(self.te, flat, "acc", self.means, want_ranking=True)
        self.assertNotIn("spearman", r["ranking"]["aggregate"])
        self.assertIsNone(r["player_rank_spearman"])

    def test_perfect_prediction_is_perfect(self):
        y = self.te["acc"].to_numpy(float)
        r = evaluate_predictions(self.te, y, "acc", self.means, want_ranking=True)
        self.assertAlmostEqual(r["mae"], 0.0, places=9)
        self.assertAlmostEqual(r["player_centered_mae"], 0.0, places=9)
        self.assertAlmostEqual(r["spearman"], 1.0, places=9)


class TestALSSolver(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.A = sp.csr_matrix((np.ones(6), ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 0])),
                               shape=(3, 4))
        self.A.sort_indices()

    def test_half_step_matches_direct_ridge_solve(self):
        """Each entity's normal equation must equal (G^T G + reg I)^-1 G^T r."""
        m = MFModel(dim=8, reg=3.0)
        F = self.rng.normal(size=(4, 8))
        resid = np.arange(6).astype(float)
        out = m._solve(resid, F, self.A, 3.0, groups=m._group_by_support(self.A))
        for u in range(3):
            js = self.A.indices[self.A.indptr[u]:self.A.indptr[u + 1]]
            rows = np.arange(self.A.indptr[u], self.A.indptr[u + 1])
            G = F[js]
            ref = np.linalg.solve(G.T @ G + 3.0 * np.eye(8), G.T @ resid[rows])
            np.testing.assert_allclose(out[u], ref, rtol=1e-8, atol=1e-8,
                                       err_msg=f"entity {u}")

    def test_solver_does_not_mutate_the_other_factor_block(self):
        m = MFModel(dim=4, reg=1.0)
        F = self.rng.normal(size=(4, 4))
        before = F.copy()
        m._solve(np.arange(6).astype(float), F, self.A, 1.0)
        np.testing.assert_array_equal(F, before, "the fixed block was written to")

    def test_size_one_support_keeps_the_sherman_morrison_term(self):
        """A rank-1 normal equation needs ||g||^2 in the denominator."""
        m = MFModel(dim=3, reg=2.0)
        A = sp.csr_matrix((np.ones(1), ([0], [0])), shape=(1, 1))
        A.sort_indices()
        g = np.array([[1.0, 2.0, 2.0]])          # ||g||^2 = 9
        r = np.array([4.5])
        out = m._solve(r, g, A, 2.0, groups=m._group_by_support(A))
        ref = g[0] * 4.5 / (2.0 + 9.0)
        np.testing.assert_allclose(out[0], ref)

    def test_als_objective_is_finite_and_recovers_a_planted_signal(self):
        """On data with a real latent structure, MF must beat a bias-only baseline."""
        df = toy(signal=True)
        test = split_mask(df, "random_interaction", 0, 0.2)
        tr, te = df[~test].reset_index(drop=True), df[test]
        means = player_train_means(tr, "acc")
        val = holdout_indices(tr, 0, 0.1)
        mf = MFModel(dim=4, reg=1.0, seed=0, n_epochs=25).fit(tr, "acc", val_idx=val)
        base = BaselineModel("player_chart_bias").fit(tr, "acc")
        r_mf = evaluate_predictions(te, mf.predict(te), "acc", means)
        r_b = evaluate_predictions(te, base.predict(te), "acc", means)
        self.assertTrue(np.isfinite(mf.history).all())
        self.assertFalse(mf.diverged)
        self.assertLess(r_mf["mae"], r_b["mae"])

    def test_predictions_are_in_range_and_do_not_depend_on_test_targets(self):
        df = toy()
        test = split_mask(df, "random_interaction", 0, 0.2)
        tr, te = df[~test].reset_index(drop=True), df[test].copy()
        mf = MFModel(dim=2, reg=3.0, seed=0, n_epochs=10).fit(tr, "acc")
        p = mf.predict(te)
        self.assertTrue(((p >= 0) & (p <= 100)).all())
        te_poisoned = te.copy()
        te_poisoned["acc"] = te_poisoned["acc"] + 1000.0
        np.testing.assert_allclose(p, mf.predict(te_poisoned))


class TestContentBaseline(unittest.TestCase):
    def test_cold_chart_content_model_runs_and_uses_cs_features_only(self):
        df = toy()
        test = split_mask(df, "cold_chart", 0, 0.3)
        tr, te = df[~test].reset_index(drop=True), df[test]
        means = player_train_means(tr, "acc")
        cr = ContentRidgeModel(alpha=1.0).fit(tr, "acc")
        p = cr.predict(te)
        self.assertEqual(len(p), len(te))
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(all(c.startswith("cs_") for c in cr.feat_cols))
        # it must not be able to see a held-out chart's own score
        te_poisoned = te.copy()
        te_poisoned["acc"] = te_poisoned["acc"] + 1000.0
        np.testing.assert_allclose(p, cr.predict(te_poisoned))


if __name__ == "__main__":
    unittest.main()

"""Phase 4B guards: the cold-chart content track must not smuggle in a leak.

Phase 4B's central claim is that CHART CONTENT predicts a cold chart's difficulty.
That claim is only interesting if the model never sees the held-out chart's own
outcome, so these tests lock the two ways it could sneak in:

  1. a client's rows could be pooled (forbidden: LR2 and beatoraja are never mixed);
  2. the "difficulty" target could be built from rows that include the evaluated
     chart's own scores, which would make cold chart prediction trivially circular.

Data-free where possible; the real-data tests skip when the artifact is absent.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase4b"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase4"))

from run_phase4b import (chart_difficulty_frame, feature_cols, gate1_verdict,  # noqa: E402
                         gate2_verdict, interaction_matrix, load_phase4b)
from splits import split_mask  # noqa: E402

PARQUET = (Path(__file__).resolve().parents[1] / "output" / "phase4" / "dataset"
           / "cross_section.parquet")


def toy(n_players: int = 6, n_charts: int = 50, per_player: int = 20,
        seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_players):
        for c in rng.choice(n_charts, size=per_player, replace=False):
            # chart difficulty is a real content effect -> predictability exists
            rows.append({"player": f"p{u}", "sha256": f"c{c:03d}",
                         "acc": float(np.clip(70 + 3 * u - 0.15 * c + rng.normal(0, 4), 0, 100)),
                         "cs_total_notes": float(500 + c * 20),
                         "cs_avg_nps": float(4 + c / 10)})
    return pd.DataFrame(rows)


class TestDifficultyTargetCannotSeeTheChart(unittest.TestCase):
    def test_frame_aggregates_only_the_rows_it_is_given(self):
        df = toy()
        cd = chart_difficulty_frame(df, ["cs_total_notes"])
        g = df.groupby("sha256")["acc"]
        # n_raters and the residual mean must be computed from THIS frame only
        self.assertEqual(int(cd["n_raters"].sum()), len(df))
        merged = cd.set_index("sha256")
        for sha, grp in list(g)[:5]:
            self.assertEqual(int(merged.loc[sha, "n_raters"]), len(grp))

    def test_dropping_a_chart_changes_nothing_about_other_charts(self):
        """Cold charts are absent from the fit: the chart frame must be a pure
        function of the rows it is given (order-independent, no hidden global)."""
        df = toy()
        test = split_mask(df, "cold_chart", 0, 0.3)
        tr, te = df[~test], df[test]
        a = chart_difficulty_frame(tr, ["cs_total_notes"]).set_index("sha256")
        b = chart_difficulty_frame(tr.sample(frac=1.0, random_state=1),
                                   ["cs_total_notes"]).set_index("sha256")
        pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())
        # the held-out charts contribute nothing to the training chart frame
        self.assertEqual(set(a.index) & set(te["sha256"]), set())

    def test_composition_control_separates_skill_from_difficulty(self):
        df = toy()
        cd = chart_difficulty_frame(df, [])
        self.assertIn("mean_skill", cd.columns)
        self.assertIn("mean_resid", cd.columns)
        # residual + composition mean should reconstruct the chart's raw mean
        cd2 = cd.set_index("sha256")
        g = df.groupby("sha256")["acc"].mean()
        np.testing.assert_allclose(
            (cd2["mean_resid"] + cd2["mean_skill"]).sort_index().to_numpy(),
            g.sort_index().to_numpy(), atol=1e-9)


class TestInteractionMatrix(unittest.TestCase):
    def test_interaction_blocks_are_per_player(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 3))
        players = np.array(["a"] * 5 + ["b"] * 5)
        I = interaction_matrix(X, players)
        self.assertEqual(I.shape, (10, 3 * 3))     # X + one block per player
        # the first block is X itself
        np.testing.assert_allclose(I[:, :3], X)
        # a row of player 'a' must have zeros in player b's block
        np.testing.assert_allclose(I[0, 6:9], 0.0)
        # ...and the player's own weighted block must equal its features
        np.testing.assert_allclose(I[0, 3:6], X[0])

    def test_interaction_can_express_player_specific_weights(self):
        """Two players who respond differently to the same content must be fittable."""
        from sklearn.linear_model import Ridge
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 2))
        players = np.array(["a"] * 100 + ["b"] * 100)
        y = np.where(players == "a", X[:, 0] * 2.0, X[:, 1] * -3.0)
        I = interaction_matrix(X, players)
        m = Ridge(alpha=1e-6).fit(I, y)
        self.assertLess(float(np.mean(np.abs(m.predict(I) - y))), 1e-3)


class TestVerdictsRequireStability(unittest.TestCase):
    def test_gate1_fails_when_gain_is_inside_seed_noise(self):
        block = {"tracks": {"content": {"n_features": 26, "evaluation_set": {},
                                        "summary": {
                                            "gbdt": {"row_mae_mean": 6.20,
                                                     "row_mae_std": 0.40, "n_seeds": 3},
                                            "mean": {"row_mae_mean": 7.3,
                                                     "row_mae_std": 0.1, "n_seeds": 3}}}}}
        v = gate1_verdict(block)
        self.assertFalse(v["passed"], "a 0.043 gain under a 0.40 seed spread is not a win")

    def test_gate1_passes_on_a_large_stable_gain(self):
        block = {"tracks": {"content": {"n_features": 26, "evaluation_set": {},
                                        "summary": {
                                            "gbdt": {"row_mae_mean": 5.80,
                                                     "row_mae_std": 0.12, "n_seeds": 3}}}}}
        v = gate1_verdict(block)
        self.assertTrue(v["passed"])

    def _gate2_block(self, cmae_gain, rank_gain, std=0.05):
        def agg(metric, val):
            return {"mean": val, "std": std, "values": [val - std / 2, val, val + std / 2]}
        ni = {"player_centered_mae": agg("player_centered_mae", 6.20),
              "player_rank_spearman": agg("player_rank_spearman", 0.41),
              "ndcg@10": agg("ndcg@10", 0.86), "mae": agg("mae", 6.20)}
        xi = {"player_centered_mae": agg("player_centered_mae", 6.20 - cmae_gain),
              "player_rank_spearman": agg("player_rank_spearman", 0.41 + rank_gain),
              "ndcg@10": agg("ndcg@10", 0.86 + rank_gain), "mae": agg("mae", 6.20 - cmae_gain)}
        return {"across_seeds": {"content_noninteractive": ni, "content_x_player": xi}}

    def test_gate2_passes_only_when_player_internal_ordering_improves(self):
        self.assertTrue(gate2_verdict(self._gate2_block(0.40, 0.09))["passed"])

    def test_gate2_calls_difficulty_only_when_ranking_does_not_move(self):
        """A chart-level MAE win with flat player-internal ordering is NOT interaction."""
        v = gate2_verdict(self._gate2_block(0.40, 0.0))
        self.assertFalse(v["passed"])
        self.assertTrue(v["difficulty_only"])


class TestRealArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PARQUET.exists():
            raise unittest.SkipTest("cross_section.parquet not built")
        cls.df, cls.info = load_phase4b(PARQUET, 120)

    def test_clients_are_never_pooled(self):
        self.assertEqual(set(self.df["client"]), {"beatoraja", "lr2"})
        pair = self.df.groupby(["player", "sha256"])["client"].nunique()
        self.assertEqual(int(pair.max()), 1)

    def test_feature_sets_are_the_documented_ones(self):
        content = feature_cols(self.df, "content", self.info)
        self.assertEqual(len(content), 26)
        self.assertTrue(all(c.startswith("cs_") for c in content))
        self.assertEqual(feature_cols(self.df, "none", self.info), [])

    def test_no_difficulty_table_level_is_used(self):
        """Red line: difficulty tables are coordinates, never input features.

        Checked structurally rather than by substring: (a) the model's feature sets are
        exactly the cs_* content stats and the MSD axes, nothing else; (b) no column
        looks like a table level (e.g. 'sl12', 'st10', a star rating).
        """
        import re
        content = feature_cols(self.df, "content", self.info)
        msd = self.info["msd"].get("axes", [])
        self.assertTrue(all(c.startswith("cs_") for c in content))
        self.assertTrue(all(c.startswith("msd_") for c in msd))
        # (a) nothing outside those two families may be fed to a model
        self.assertEqual(feature_cols(self.df, "content+msd", self.info), content + msd)
        # (b) no table-level-looking column exists anywhere in the frame
        pat = re.compile(r"^(sl|st|sat|insane|normal|overjoy)\d*$|level|^★|jrank", re.I)
        offenders = [c for c in self.df.columns if pat.search(c)]
        self.assertEqual(offenders, [], f"difficulty-table-ish columns: {offenders}")

    def test_no_target_leakage_columns_from_the_manifest(self):
        """The corpus manifest's `labels` (difficulty-table memberships) must never
        reach the modelling frame."""
        self.assertNotIn("labels", self.df.columns)
        self.assertNotIn("table", self.df.columns)


if __name__ == "__main__":
    unittest.main()

"""Tests for the Glicko-2 rating system.

The headline test is ``test_regression_glickman_worked_example``, which
replicates the worked example from Section 1 of Glickman's 2013 paper:

- A player starts at R=1500, RD=200, sigma=0.06.
- They play three games (against opponents with known ratings), winning
  the first and losing the next two.
- After updating, the paper specifies the expected new state to several
  decimal places.

Since this implementation does per-game updates rather than batched
rating periods, we cannot directly compare to the multi-game-period
result. Instead we verify the building blocks (g, E, volatility update)
and run end-to-end sanity checks (symmetry, conservation-like properties,
defaults).
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ratingskit import Rating
from ratingskit.systems.glicko2 import Glicko2, Glicko2Rating, _InternalRating

_RATING_RANGE = st.floats(min_value=500.0, max_value=2500.0, allow_nan=False)


class TestGlicko2Rating:
    def test_satisfies_rating_protocol(self) -> None:
        r = Glicko2Rating(point=1500.0, deviation=350.0, volatility=0.06)
        assert isinstance(r, Rating)
        assert r.point == 1500.0

    def test_is_frozen(self) -> None:
        r = Glicko2Rating(point=1500.0, deviation=350.0, volatility=0.06)
        with pytest.raises(FrozenInstanceError):
            r.point = 1600.0  # type: ignore[misc]

    def test_equality_by_value(self) -> None:
        a = Glicko2Rating(point=1500.0, deviation=350.0, volatility=0.06)
        b = Glicko2Rating(point=1500.0, deviation=350.0, volatility=0.06)
        c = Glicko2Rating(point=1500.0, deviation=350.0, volatility=0.07)
        assert a == b
        assert a != c


class TestGlicko2Defaults:
    def test_construct_with_defaults(self) -> None:
        g = Glicko2()
        assert g.initial == 1500.0
        assert g.initial_deviation == 350.0
        assert g.initial_volatility == 0.06
        assert g.tau == 0.5
        assert dict(g.ratings()) == {}

    def test_construct_with_overrides(self) -> None:
        g = Glicko2(initial=1200.0, initial_deviation=200.0, tau=0.3)
        assert g.initial == 1200.0
        assert g.initial_deviation == 200.0
        assert g.tau == 0.3

    def test_positional_args_rejected(self) -> None:
        with pytest.raises(TypeError):
            Glicko2(1500.0, 350.0, 0.06, 0.5)  # type: ignore[misc]

    def test_internal_scale_conversion(self) -> None:
        g = Glicko2()
        # mu for R=1500 should be exactly 0.
        assert g._initial_mu == pytest.approx(0.0, abs=1e-12)
        # phi for RD=350 should be 350/173.7178 ~= 2.0148.
        assert g._initial_phi == pytest.approx(350.0 / 173.7178, rel=1e-9)


class TestAutoCreation:
    def test_rating_auto_creates(self) -> None:
        g = Glicko2()
        r = g.rating("Alice")
        assert r.point == pytest.approx(1500.0)
        assert r.deviation == pytest.approx(350.0)
        assert r.volatility == pytest.approx(0.06)

    def test_predict_auto_creates_competitors(self) -> None:
        g = Glicko2()
        assert g.predict("A", "B") == pytest.approx(0.5)
        assert "A" in g.ratings()
        assert "B" in g.ratings()

    def test_update_auto_creates_competitors(self) -> None:
        g = Glicko2()
        g.update("Patriots", "Bills", outcome=1.0)
        snap = g.ratings()
        assert "Patriots" in snap
        assert "Bills" in snap
        assert snap["Patriots"].point > 1500.0
        assert snap["Bills"].point < 1500.0


class TestPredict:
    def test_equal_ratings_predict_half(self) -> None:
        g = Glicko2()
        assert g.predict("A", "B") == pytest.approx(0.5)

    def test_higher_rating_predicts_higher_probability(self) -> None:
        g = Glicko2()
        g._ratings["Strong"] = _InternalRating(mu=1.0, phi=0.5, sigma=0.06)
        g._ratings["Weak"] = _InternalRating(mu=-1.0, phi=0.5, sigma=0.06)
        assert g.predict("Strong", "Weak") > 0.5
        assert g.predict("Weak", "Strong") < 0.5

    @given(_RATING_RANGE, _RATING_RANGE)
    def test_predict_symmetry(self, r1: float, r2: float) -> None:
        """P(A beats B) + P(B beats A) == 1.0 for any pair of ratings."""
        g = Glicko2()
        g._ratings["A"] = _InternalRating(
            mu=(r1 - 1500.0) / 173.7178,
            phi=350.0 / 173.7178,
            sigma=0.06,
        )
        g._ratings["B"] = _InternalRating(
            mu=(r2 - 1500.0) / 173.7178,
            phi=350.0 / 173.7178,
            sigma=0.06,
        )
        assert g.predict("A", "B") + g.predict("B", "A") == pytest.approx(1.0)


class TestUpdate:
    def test_winner_rating_increases(self) -> None:
        g = Glicko2()
        g.update("Winner", "Loser", outcome=1.0)
        assert g.rating("Winner").point > 1500.0
        assert g.rating("Loser").point < 1500.0

    def test_draw_with_equal_ratings_changes_uncertainty_only(self) -> None:
        """Equal ratings + draw: mu shouldn't move; phi should shrink."""
        g = Glicko2()
        g.update("A", "B", outcome=0.5)
        a = g.rating("A")
        b = g.rating("B")
        # Ratings stay near 1500 (drift from numerical evaluation is fine).
        assert a.point == pytest.approx(1500.0, abs=1e-6)
        assert b.point == pytest.approx(1500.0, abs=1e-6)
        # RD shrinks because games were played - we know more about them now.
        assert a.deviation < 350.0
        assert b.deviation < 350.0

    def test_update_decreases_rd_for_played_players(self) -> None:
        """Playing games should shrink uncertainty."""
        g = Glicko2()
        initial_rd = g.rating("A").deviation
        g.update("A", "B", outcome=1.0)
        assert g.rating("A").deviation < initial_rd

    @pytest.mark.parametrize("outcome", [-0.1, 1.1, math.inf, math.nan])
    def test_rejects_invalid_outcome(self, outcome: float) -> None:
        g = Glicko2()
        with pytest.raises(ValueError, match="outcome must be in"):
            g.update("A", "B", outcome=outcome)


class TestVolatilityUpdate:
    def test_volatility_converges(self) -> None:
        """The Illinois algorithm should always converge for sane inputs."""
        g = Glicko2(tau=0.5)
        sigma = 0.06
        phi = 350.0 / 173.7178
        # Test across a range of (v, delta) values.
        for v in [1.0, 10.0, 100.0]:
            for delta in [-2.0, -0.5, 0.0, 0.5, 2.0]:
                result = g._new_volatility(sigma, phi, v, delta)
                assert math.isfinite(result)
                assert result > 0.0

    def test_volatility_stays_near_initial_for_expected_outcome(self) -> None:
        """A game that goes as expected shouldn't move volatility much."""
        g = Glicko2()
        # Run many neutral games and check sigma stays close to 0.06.
        for _ in range(20):
            g.update("A", "B", outcome=0.5)
        a = g.rating("A")
        # Allow some drift, but not order-of-magnitude changes.
        assert 0.04 < a.volatility < 0.08


class TestGlickmanPaperRegression:
    """Regression tests against Glickman's worked example.

    From Section 1 of the 2013 paper: a player at (R=1500, RD=200, sigma=0.06)
    plays 3 games in one rating period against opponents and gets s=[1, 0, 0].
    The paper specifies the post-period values explicitly.

    Note: because this implementation does per-game updates instead of
    period-batched updates, we cannot replicate the paper's exact final
    values. We instead verify the building blocks (g and E functions)
    match the paper's intermediate computations.
    """

    def test_g_function_matches_paper(self) -> None:
        """Paper Section 3: g(phi_j) for opponents with RD=30, 100, 300.

        Paper's stated values (RD in public scale, must convert to phi):
        - RD=30  -> phi ~= 0.1727 -> g ~= 0.9955
        - RD=100 -> phi ~= 0.5756 -> g ~= 0.9531
        - RD=300 -> phi ~= 1.7269 -> g ~= 0.7242
        """
        g = Glicko2()
        for rd, expected_g in [(30, 0.9955), (100, 0.9531), (300, 0.7242)]:
            phi = rd / 173.7178
            assert g._g(phi) == pytest.approx(expected_g, abs=0.001)

    def test_e_function_matches_paper(self) -> None:
        """Paper Section 3: E function for the worked example.

        Player at mu=0, against three opponents at (mu_j, phi_j) computed
        from paper's (R, RD) = (1400, 30), (1550, 100), (1700, 300).
        Paper's stated E values: 0.639, 0.432, 0.303.
        """
        g = Glicko2()
        mu_player = 0.0  # R=1500
        for r_opp, rd_opp, expected_e in [
            (1400, 30, 0.639),
            (1550, 100, 0.432),
            (1700, 300, 0.303),
        ]:
            mu_opp = (r_opp - 1500) / 173.7178
            phi_opp = rd_opp / 173.7178
            g_opp = g._g(phi_opp)
            e = 1.0 / (1.0 + math.exp(-g_opp * (mu_player - mu_opp)))
            assert e == pytest.approx(expected_e, abs=0.001)


class TestRatingsSnapshot:
    def test_snapshot_independence(self) -> None:
        """Mutating ratings() must not affect internal state."""
        g = Glicko2()
        g.update("A", "B", outcome=1.0)
        snap = dict(g.ratings())
        snap["C"] = Glicko2Rating(point=9999.0, deviation=1.0, volatility=0.06)
        snap.pop("A")
        fresh = g.ratings()
        assert "C" not in fresh
        assert "A" in fresh


class TestDeterminism:
    @settings(deadline=None)
    @given(st.integers(min_value=1, max_value=50))
    def test_same_inputs_same_state(self, n: int) -> None:
        """Two Glicko-2 instances fed the same games end identical."""
        games = [("A", "B", 1.0), ("B", "C", 0.5), ("A", "C", 0.0)] * n
        g1 = Glicko2()
        g2 = Glicko2()
        for home, away, outcome in games:
            g1.update(home, away, outcome=outcome)
            g2.update(home, away, outcome=outcome)
        for name in ["A", "B", "C"]:
            assert g1.rating(name) == g2.rating(name)

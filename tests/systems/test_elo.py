"""Tests for the Elo rating system.

These tests cover:
- The ``EloRating`` value type and its protocol conformance.
- ``Elo`` construction defaults and auto-creation behavior.
- The Elo math (predict symmetry, conservation, regression against a
  hand-computed reference).
- The ``ratings()`` snapshot's defensive copy semantics.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ratingskit import Rating
from ratingskit.systems.elo import Elo, EloRating

# Reasonable bounds for Hypothesis - cover the full range of plausible Elo
# ratings without straying into floating-point overflow territory.
_RATING_RANGE = st.floats(min_value=100.0, max_value=3000.0, allow_nan=False)


class TestEloRating:
    def test_satisfies_rating_protocol(self):
        r = EloRating(1500.0)
        assert isinstance(r, Rating)
        assert r.point == 1500.0

    def test_is_frozen(self):
        r = EloRating(1500.0)
        with pytest.raises(FrozenInstanceError):
            r.point = 1600.0  # type: ignore[misc]

    def test_equality_by_value(self):
        assert EloRating(1500.0) == EloRating(1500.0)
        assert EloRating(1500.0) != EloRating(1500.1)

    def test_hashable(self):
        # Frozen dataclasses are hashable; should work as a dict key.
        d = {EloRating(1500.0): "home"}
        assert d[EloRating(1500.0)] == "home"


class TestEloDefaults:
    def test_construct_with_defaults(self):
        elo = Elo()
        assert elo.k == 20.0
        assert elo.initial == 1500.0
        assert elo.home_advantage == 0.0
        assert elo.ratings() == {}

    def test_construct_with_overrides(self):
        elo = Elo(k=32.0, initial=1000.0, home_advantage=55.0)
        assert elo.k == 32.0
        assert elo.initial == 1000.0
        assert elo.home_advantage == 55.0

    def test_positional_args_rejected(self):
        # The * in the signature forces keyword-only construction. This
        # prevents Elo(20, 1500, 55) from accidentally swapping K and HFA.
        with pytest.raises(TypeError):
            Elo(20.0, 1500.0, 0.0)  # type: ignore[misc]


class TestAutoCreation:
    def test_predict_auto_creates_competitors(self):
        elo = Elo()
        # No update has happened. Equal ratings should give 0.5.
        assert elo.predict("A", "B") == pytest.approx(0.5)
        # Both should now exist at the initial rating.
        assert elo.ratings()["A"].point == 1500.0
        assert elo.ratings()["B"].point == 1500.0

    def test_rating_auto_creates_competitor(self):
        elo = Elo(initial=1200.0)
        assert elo.rating("Alice").point == 1200.0
        assert "Alice" in elo.ratings()

    def test_update_auto_creates_competitors(self):
        elo = Elo()
        elo.update("Patriots", "Bills", outcome=1.0)
        # Both teams now exist with diverged ratings.
        snapshot = elo.ratings()
        assert "Patriots" in snapshot
        assert "Bills" in snapshot
        assert snapshot["Patriots"].point > 1500.0
        assert snapshot["Bills"].point < 1500.0


class TestEloMath:
    def test_equal_ratings_predict_half(self):
        elo = Elo()
        assert elo.predict("A", "B") == pytest.approx(0.5)

    def test_home_advantage_shifts_prediction_upward(self):
        neutral = Elo(home_advantage=0.0)
        with_hfa = Elo(home_advantage=100.0)
        # Same matchup, HFA only changes the prediction.
        assert with_hfa.predict("A", "B") > neutral.predict("A", "B")
        assert with_hfa.predict("A", "B") > 0.5

    @given(_RATING_RANGE, _RATING_RANGE)
    def test_predict_symmetry_without_hfa(self, r1: float, r2: float):
        """P(A beats B) + P(B beats A) == 1.0 when HFA is zero."""
        elo = Elo(home_advantage=0.0)
        # Seed two competitors at specific ratings via private state.
        # Using public API would require update() calls which would mutate.
        elo._ratings["A"] = EloRating(r1)
        elo._ratings["B"] = EloRating(r2)
        assert elo.predict("A", "B") + elo.predict("B", "A") == pytest.approx(1.0)

    def test_winner_gains_what_loser_loses(self):
        """Zero-sum conservation: the rating pool's total is preserved."""
        elo = Elo(k=32.0)
        elo.update("A", "B", outcome=1.0)
        snap = elo.ratings()
        gain = snap["A"].point - 1500.0
        loss = 1500.0 - snap["B"].point
        assert gain == pytest.approx(loss)

    def test_draw_with_equal_ratings_is_a_noop(self):
        """A draw between equal-rated competitors shouldn't change anything."""
        elo = Elo()
        elo.update("A", "B", outcome=0.5)
        snap = elo.ratings()
        assert snap["A"].point == pytest.approx(1500.0)
        assert snap["B"].point == pytest.approx(1500.0)

    def test_regression_known_values(self):
        """Hand-computed reference: 1500 beats 1500 with K=32, HFA=0.

        Expected score for equal ratings = 0.5.
        Delta = 32 * (1.0 - 0.5) = 16.0.
        Winner: 1500 + 16 = 1516. Loser: 1500 - 16 = 1484.
        """
        elo = Elo(k=32.0, home_advantage=0.0)
        elo.update("A", "B", outcome=1.0)
        snap = elo.ratings()
        assert snap["A"].point == pytest.approx(1516.0)
        assert snap["B"].point == pytest.approx(1484.0)

    def test_regression_400_point_gap(self):
        """A 400-point gap means the favorite has ~91% win probability.

        Per Elo's definition, a 400-point gap corresponds to 10:1 odds,
        i.e. P(favorite wins) = 10/11 ~= 0.909.
        """
        elo = Elo(home_advantage=0.0)
        elo._ratings["Strong"] = EloRating(1900.0)
        elo._ratings["Weak"] = EloRating(1500.0)
        assert elo.predict("Strong", "Weak") == pytest.approx(10.0 / 11.0, abs=1e-9)


class TestUpdateValidation:
    @pytest.mark.parametrize("outcome", [-0.1, 1.1, -1.0, 2.0, math.inf, math.nan])
    def test_rejects_invalid_outcome(self, outcome: float):
        elo = Elo()
        with pytest.raises(ValueError, match="outcome must be in"):
            elo.update("A", "B", outcome=outcome)

    @pytest.mark.parametrize("outcome", [0.0, 0.5, 1.0])
    def test_accepts_valid_outcome(self, outcome: float):
        elo = Elo()
        elo.update("A", "B", outcome=outcome)  # no raise


class TestRatingsSnapshot:
    def test_returns_defensive_copy(self):
        """Mutating the returned dict must not affect internal state."""
        elo = Elo()
        elo.update("A", "B", outcome=1.0)
        snap = elo.ratings()
        snap["C"] = EloRating(9999.0)
        snap.pop("A")
        # Internal state untouched.
        fresh = elo.ratings()
        assert "C" not in fresh
        assert "A" in fresh


class TestDeterminism:
    def test_same_inputs_produce_same_state(self):
        """Two Elo instances fed the same games end in identical states."""
        games = [
            ("A", "B", 1.0),
            ("B", "C", 0.5),
            ("A", "C", 0.0),
            ("C", "A", 1.0),
        ]
        elo1 = Elo(k=24.0)
        elo2 = Elo(k=24.0)
        for home, away, outcome in games:
            elo1.update(home, away, outcome=outcome)
            elo2.update(home, away, outcome=outcome)
        assert elo1.ratings() == elo2.ratings()

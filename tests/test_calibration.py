"""Tests for the calibration module."""

from __future__ import annotations

import math

import pytest

from ratingskit import (
    CoinFlip,
    Elo,
    Game,
    Predictor,
    accuracy,
    brier_score,
    log_loss,
)


class AlwaysHome:
    """Predictor that always says home wins with high confidence."""

    def predict(self, home: str, away: str) -> float:  # noqa: ARG002
        return 0.9


class ExtremePredictor:
    """Predictor that returns a fixed value, useful for boundary tests."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, home: str, away: str) -> float:  # noqa: ARG002
        return self.value


class TestBrierScore:
    def test_neutral_predictions_against_mixed_outcomes(self) -> None:
        """Predictor returning 0.5 against mixed outcomes scores 0.25.

        Math: (0.5 - 1.0)^2 = 0.25, (0.5 - 0.0)^2 = 0.25, average = 0.25.
        """
        games = [
            Game(home="A", away="B", outcome=1.0),
            Game(home="C", away="D", outcome=0.0),
        ]
        assert brier_score(Elo(), games) == pytest.approx(0.25)

    def test_perfect_predictor_scores_zero(self) -> None:
        games = [Game(home="A", away="B", outcome=1.0)]
        perfect = ExtremePredictor(1.0)
        assert brier_score(perfect, games) == pytest.approx(0.0)

    def test_worst_predictor_scores_one(self) -> None:
        games = [Game(home="A", away="B", outcome=0.0)]
        wrong = ExtremePredictor(1.0)
        assert brier_score(wrong, games) == pytest.approx(1.0)

    def test_empty_games_raises(self) -> None:
        with pytest.raises(ValueError, match="games must be non-empty"):
            brier_score(Elo(), [])


class TestLogLoss:
    def test_neutral_predictions_score_ln_two(self) -> None:
        """Predictor returning 0.5 has log loss equal to ln(2)."""
        games = [
            Game(home="A", away="B", outcome=1.0),
            Game(home="C", away="D", outcome=0.0),
        ]
        assert log_loss(Elo(), games) == pytest.approx(math.log(2.0))

    def test_clipping_prevents_log_zero_crash(self) -> None:
        """A prediction of exactly 1.0 on a losing outcome would compute
        log(0) without clipping. The result should be very high but finite.
        """
        games = [Game(home="A", away="B", outcome=0.0)]
        result = log_loss(ExtremePredictor(1.0), games)
        assert math.isfinite(result)
        assert result > 30

    def test_empty_games_raises(self) -> None:
        with pytest.raises(ValueError, match="games must be non-empty"):
            log_loss(Elo(), [])


class TestAccuracy:
    def test_perfect_alignment_scores_one(self) -> None:
        games = [
            Game(home="A", away="B", outcome=1.0),
            Game(home="C", away="D", outcome=1.0),
        ]
        assert accuracy(AlwaysHome(), games) == pytest.approx(1.0)

    def test_perfect_misalignment_scores_zero(self) -> None:
        games = [
            Game(home="A", away="B", outcome=0.0),
            Game(home="C", away="D", outcome=0.0),
        ]
        assert accuracy(AlwaysHome(), games) == pytest.approx(0.0)

    def test_neutral_predictor_with_mixed_outcomes(self) -> None:
        """A predictor returning exactly 0.5 does not exceed the 0.5 threshold,
        so it classifies every game as 'away wins'. On a 50/50 mix of outcomes
        it is right on the away-wins games only.
        """
        games = [
            Game(home="A", away="B", outcome=1.0),
            Game(home="C", away="D", outcome=0.0),
        ]
        assert accuracy(Elo(), games) == pytest.approx(0.5)

    def test_custom_threshold(self) -> None:
        games = [Game(home="A", away="B", outcome=1.0)]
        assert accuracy(AlwaysHome(), games, threshold=0.7) == pytest.approx(1.0)
        assert accuracy(AlwaysHome(), games, threshold=0.95) == pytest.approx(0.0)

    def test_empty_games_raises(self) -> None:
        with pytest.raises(ValueError, match="games must be non-empty"):
            accuracy(Elo(), [])


class TestCoinFlip:
    def test_always_returns_half(self) -> None:
        cf = CoinFlip()
        assert cf.predict("anything", "anything else") == 0.5
        assert cf.predict("Patriots", "Bills") == 0.5

    def test_satisfies_predictor_protocol(self) -> None:
        assert isinstance(CoinFlip(), Predictor)

    def test_works_with_calibration_functions(self) -> None:
        games = [
            Game(home="A", away="B", outcome=1.0),
            Game(home="C", away="D", outcome=0.0),
        ]
        assert brier_score(CoinFlip(), games) == pytest.approx(0.25)
        assert log_loss(CoinFlip(), games) == pytest.approx(math.log(2.0))

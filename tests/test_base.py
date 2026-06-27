"""Tests for the base protocols and abstract classes in ratingskit.base."""

from __future__ import annotations

from ratingskit import Elo, Glicko2, Predictor


class TestPredictorProtocol:
    def test_elo_satisfies_predictor(self) -> None:
        elo = Elo()
        assert isinstance(elo, Predictor)

    def test_glicko2_satisfies_predictor(self) -> None:
        g = Glicko2()
        assert isinstance(g, Predictor)

"""Smoke tests for the base module. Replace as real systems are implemented."""

from __future__ import annotations

from datetime import date

import pytest

from ratingskit import Game, Rating, RatingSystem


class TestGame:
    def test_construct_minimal(self):
        g = Game(home="A", away="B", outcome=1.0)
        assert g.home == "A"
        assert g.away == "B"
        assert g.outcome == 1.0
        assert g.date is None
        assert g.home_score is None
        assert g.away_score is None

    def test_construct_full(self):
        g = Game(
            home="A",
            away="B",
            outcome=1.0,
            date=date(2024, 1, 1),
            home_score=24,
            away_score=17,
        )
        assert g.date == date(2024, 1, 1)
        assert g.home_score == 24
        assert g.away_score == 17

    def test_is_frozen(self):
        g = Game(home="A", away="B", outcome=1.0)
        with pytest.raises(AttributeError):
            g.home = "C"  # type: ignore[misc]


class TestRatingProtocol:
    def test_satisfies_protocol(self):
        class Concrete:
            @property
            def point(self) -> float:
                return 1500.0

        assert isinstance(Concrete(), Rating)


class TestRatingSystemABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            RatingSystem()  # type: ignore[abstract]

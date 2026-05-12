"""Base types, protocols, and abstract classes for rating systems.

This module defines the contract every rating system in ratingskit must satisfy.
Implementations live in ``ratingskit.systems``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import date

    import pandas as pd


@dataclass(frozen=True, slots=True)
class Game:
    """A single completed game between two competitors.

    Attributes:
        home: Identifier for the home/first competitor.
        away: Identifier for the away/second competitor.
        outcome: 1.0 if home wins, 0.0 if away wins, 0.5 for a draw.
        date: Optional date the game was played. Used by time-aware systems
            and modifiers (e.g. ``TimeDecay``).
        home_score: Optional score for the home competitor. Used by
            margin-of-victory adjustments.
        away_score: Optional score for the away competitor.
    """

    home: str
    away: str
    outcome: float
    date: date | None = None
    home_score: float | None = None
    away_score: float | None = None


@runtime_checkable
class Rating(Protocol):
    """A competitor's rating under some system.

    Implementations may carry additional fields (uncertainty, volatility, etc.),
    but every rating must expose a scalar point estimate via ``point`` so that
    cross-system tooling (calibration, plotting, comparison) can operate
    uniformly.
    """

    @property
    def point(self) -> float:
        """Scalar point estimate of this competitor's strength."""
        ...


class RatingSystem(ABC):
    """Abstract base class for all rating systems.

    Subclasses implement ``update``, ``predict``, ``rating``, and ``ratings``.
    The default ``fit`` iterates ``update`` in order. Systems with closed-form
    batch solutions (Massey, Colley) should override ``fit``.
    """

    @abstractmethod
    def update(
        self,
        home: str,
        away: str,
        outcome: float,
        *,
        date: date | None = None,
        home_score: float | None = None,
        away_score: float | None = None,
    ) -> None:
        """Apply a single game result, mutating internal state.

        Args:
            home: Identifier for the home competitor.
            away: Identifier for the away competitor.
            outcome: 1.0 home win, 0.0 away win, 0.5 draw.
            date: Optional game date.
            home_score: Optional home score (for MOV-aware systems).
            away_score: Optional away score (for MOV-aware systems).
        """

    @abstractmethod
    def predict(self, home: str, away: str) -> float:
        """Return the probability that ``home`` beats ``away``.

        Must lie in ``[0, 1]``. Implementations should be deterministic for
        a given internal state.
        """

    @abstractmethod
    def rating(self, competitor: str) -> Rating:
        """Return the current rating for ``competitor``.

        Implementations should return a system-appropriate ``Rating`` instance
        (e.g. a scalar wrapper for Elo, a ``(mu, phi)`` pair for Glicko-2).
        """

    @abstractmethod
    def ratings(self) -> dict[str, Rating]:
        """Return a snapshot of all current ratings keyed by competitor."""

    def fit(self, games: pd.DataFrame | Iterable[Game]) -> None:
        """Apply a sequence of games in order.

        Default implementation iterates ``update`` once per game. Systems with
        batch or matrix-based solutions should override this.

        Args:
            games: Either a pandas DataFrame with columns
                ``home, away, outcome`` (and optionally ``date``,
                ``home_score``, ``away_score``), or an iterable of ``Game``.
        """
        # Local import to keep pandas out of the hot path for users who only
        # call ``update`` directly.
        import pandas as pd  # noqa: PLC0415

        if isinstance(games, pd.DataFrame):
            for row in games.itertuples(index=False):
                self.update(
                    home=cast("str", row.home),
                    away=cast("str", row.away),
                    outcome=cast("float", row.outcome),
                    date=cast("date | None", getattr(row, "date", None)),
                    home_score=cast("float | None", getattr(row, "home_score", None)),
                    away_score=cast("float | None", getattr(row, "away_score", None)),
                )
        else:
            for game in games:
                self.update(
                    home=game.home,
                    away=game.away,
                    outcome=game.outcome,
                    date=game.date,
                    home_score=game.home_score,
                    away_score=game.away_score,
                )

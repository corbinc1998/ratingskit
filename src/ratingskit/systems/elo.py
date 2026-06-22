"""Elo rating system.

Reference: Arpad Elo, *The Rating of Chessplayers, Past and Present* (1978).
The expected-score formula used here is the standard logistic form on base 10
with a scale of 400 points per factor-of-10 odds shift::

    E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))

Rating updates follow::

    R_A' = R_A + K * (S_A - E_A)

where S_A is the actual outcome (1.0 win, 0.5 draw, 0.0 loss) for player A.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ratingskit.base import RatingSystem

if TYPE_CHECKING:
    from datetime import date


# Elo's scale constant. Hardcoded here rather than parameterized because changing
# it would break the meaning of "Elo rating" - it is part of the system's definition.
# A scale of 400 means a 400-point gap implies the higher-rated player is 10x as
# likely to win.
_SCALE = 400.0


@dataclass(frozen=True, slots=True)
class EloRating:
    """An Elo rating: a single scalar strength estimate."""

    point: float


class Elo(RatingSystem):
    """Classic Elo rating system.

    Args:
        k: K-factor. Controls how much a single result moves ratings.
            Common values: 16 (settled players), 24 (default for many systems),
            32 (FIDE for unrated/junior players), 20 (FiveThirtyEight NFL Elo).
        initial: Rating assigned to a competitor on first appearance.
        home_advantage: Points added to the home competitor's rating during
            prediction and expected-score calculation. Not stored on the
            rating itself - the rating represents true strength; HFA is a
            contextual modifier. Set to 0 for neutral-site systems (chess,
            most esports). Common NFL values: 55-65.

    Example:
        >>> elo = Elo(k=20, initial=1500, home_advantage=55)
        >>> elo.update("Patriots", "Bills", outcome=1.0)
        >>> elo.predict("Patriots", "Bills")  # doctest: +ELLIPSIS
        0.6...
    """

    def __init__(
        self,
        *,
        k: float = 20.0,
        initial: float = 1500.0,
        home_advantage: float = 0.0,
    ) -> None:
        self.k: float = k
        self.initial: float = initial
        self.home_advantage: float = home_advantage
        self._ratings: dict[str, EloRating] = {}

    # ------------------------------------------------------------------
    # Public API (required by RatingSystem)
    # ------------------------------------------------------------------

    def update(
        self,
        home: str,
        away: str,
        outcome: float,
        *,
        date: date | None = None,  # noqa: ARG002 - part of the base contract
        home_score: float | None = None,  # noqa: ARG002 - used by MOV modifiers
        away_score: float | None = None,  # noqa: ARG002 - used by MOV modifiers
    ) -> None:
        """Apply a single game result.

        Args:
            home: Identifier for the home competitor.
            away: Identifier for the away competitor.
            outcome: 1.0 if home wins, 0.0 if away wins, 0.5 for a draw.
            date: Ignored by base Elo. Present for protocol compatibility
                and used by time-decay modifiers.
            home_score: Ignored by base Elo. Used by MOV-adjusted variants.
            away_score: Ignored by base Elo. Used by MOV-adjusted variants.

        Raises:
            ValueError: If ``outcome`` is outside [0.0, 1.0].
        """
        if not 0.0 <= outcome <= 1.0:
            msg = f"outcome must be in [0.0, 1.0], got {outcome}"
            raise ValueError(msg)

        r_home = self._get_or_create(home)
        r_away = self._get_or_create(away)

        expected_home = self._expected(r_home, r_away)
        delta = self._k_for(home) * (outcome - expected_home)

        # Zero-sum: home gains delta, away loses the same. This conservation
        # property is what keeps the rating pool's mean stable over time.
        self._ratings[home] = EloRating(r_home + delta)
        self._ratings[away] = EloRating(r_away - delta)

    def predict(self, home: str, away: str) -> float:
        """Return the probability that ``home`` beats ``away``.

        Auto-creates either competitor at ``initial`` if not yet seen, so
        callers can ask for predictions before any games have been recorded.
        """
        r_home = self._get_or_create(home)
        r_away = self._get_or_create(away)
        return self._expected(r_home, r_away)

    def rating(self, competitor: str) -> EloRating:
        """Return the current rating for ``competitor``.

        Auto-creates the competitor at ``initial`` if not yet seen.
        """
        return EloRating(self._get_or_create(competitor))

    def ratings(self) -> dict[str, EloRating]:
        """Return a snapshot of all current ratings.

        The returned dict is a copy - mutating it does not affect internal
        state.
        """
        return dict(self._ratings)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_or_create(self, name: str) -> float:
        """Return the float rating for ``name``, creating it if absent.

        This is the single point where unknown competitors are auto-registered.
        Centralizing the logic here means ``update``, ``predict``, and
        ``rating`` all have consistent first-time-seen behavior.
        """
        existing = self._ratings.get(name)
        if existing is None:
            self._ratings[name] = EloRating(self.initial)
            return self.initial
        return existing.point

    def _expected(self, r_home: float, r_away: float) -> float:
        """Expected score for home, with home advantage folded in.

        Home advantage is added to the home rating *only inside this
        computation*. It does not mutate the stored rating - storing it
        would cause drift across long histories.
        """
        diff = (r_away - (r_home + self.home_advantage)) / _SCALE
        return 1.0 / (1.0 + math.pow(10.0, diff))

    def _k_for(self, competitor: str) -> float:  # noqa: ARG002
        """Effective K-factor for a competitor.

        Currently returns the system-wide K. Exists as a hook for
        rating-dependent K schedules (FIDE's tiered K, provisional ratings,
        etc.) without touching the update equation.
        """
        return self.k

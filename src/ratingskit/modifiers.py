from __future__ import annotations

from datetime import date as _date
from ratingskit.base import RatingSystem
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Mapping

    from ratingskit.base import Rating

class TimeDecay(RatingSystem):
    def __init__(self, wrapped: RatingSystem, *, half_life_days: float) -> None:
        self.wrapped: RatingSystem = wrapped
        self.half_life_days: float = half_life_days

    def predict(self, home: str, away: str) -> float:
        return self.wrapped.predict(home, away)

    def rating(self, competitor: str) -> Rating:
        return self.wrapped.rating(competitor)

    def ratings(self) -> Mapping[str, Rating]:
        return self.wrapped.ratings()

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
        if date is None:
            raise ValueError
        


    
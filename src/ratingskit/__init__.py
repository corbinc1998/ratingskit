"""ratingskit: a unified, composable library for competitive rating systems."""

from ratingskit.base import Game, Rating, RatingSystem
from ratingskit.systems.elo import Elo, EloRating

__version__ = "0.0.1"

__all__ = [
    "Elo",
    "EloRating",
    "Game",
    "Rating",
    "RatingSystem",
    "__version__",
]

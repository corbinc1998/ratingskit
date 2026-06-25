"""ratingskit: a unified, composable library for competitive rating systems."""

from ratingskit.base import Game, Rating, RatingSystem
from ratingskit.systems.elo import Elo, EloRating
from ratingskit.systems.glicko2 import Glicko2, Glicko2Rating

__version__ = "0.0.1"

__all__ = [
    "Elo",
    "EloRating",
    "Game",
    "Glicko2",
    "Glicko2Rating",
    "Rating",
    "RatingSystem",
    "__version__",
]

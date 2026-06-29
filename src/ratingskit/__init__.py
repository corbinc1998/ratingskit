"""ratingskit: a unified, composable library for competitive rating systems."""

from ratingskit.base import Game, Predictor, Rating, RatingSystem
from ratingskit.calibration import CoinFlip, accuracy, brier_score, log_loss
from ratingskit.systems.elo import Elo, EloRating
from ratingskit.systems.glicko2 import Glicko2, Glicko2Rating

__version__ = "0.0.1"

__all__ = [
    "CoinFlip",
    "Elo",
    "EloRating",
    "Game",
    "Glicko2",
    "Glicko2Rating",
    "Predictor",
    "Rating",
    "RatingSystem",
    "__version__",
    "accuracy",
    "brier_score",
    "log_loss",
]

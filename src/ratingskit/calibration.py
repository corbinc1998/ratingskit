from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ratingskit.base import Game, Predictor


def brier_score(predictor: Predictor, games: list[Game]) -> float:
    if not games:
        raise ValueError("games must be non-empty")
    total = 0
    squared_difference = 0.0
    for game in games:
        prediction = predictor.predict(game.home, game.away)
        squared_difference += (prediction - game.outcome) ** 2
        total += 1
    return squared_difference / total

def log_loss(predictor: Predictor, games: list[Game]) -> float:
    total = 0
    loss = 0.0
    if not games:
        raise ValueError("games must be non-empty")
    for game in games:
        total += 1
        prediction = predictor.predict(game.home, game.away)
        prediction = max(1e-15, min(prediction, 1 - 1e-15))
        loss += -(
            game.outcome * math.log(prediction) + (1 - game.outcome) * math.log(1 - prediction)
            )
    return loss /total

def accuracy(predictor: Predictor, games: list[Game], threshold: float = 0.5) -> float:
    if not games:
        raise ValueError("games must be non-empty")
    correct = 0
    total = len(games)
    for game in games:
        prediction = predictor.predict(game.home, game.away)
        predicted_home_wins = prediction > threshold
        actually_home_won = game.outcome > 0.5  
        if predicted_home_wins == actually_home_won:
            correct += 1
    return correct / total



from __future__ import annotations
from ratingskit.base import Game, Predictor


# (a - b) ** 2
def brier_score(predictor: Predictor, games: list[Game]) -> float:
    prediction = predictor.predict(game.home, game.away)
    total = 0
    squared_difference = 0
    for game in games:
       prediction = predictor.predict(game.home, game.away)
       squared_difference += (prediction - game.outcome) **2
       total +=1
       score = squared_difference /total

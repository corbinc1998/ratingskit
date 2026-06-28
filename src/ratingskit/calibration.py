from __future__ import annotations
from ratingskit.base import Game, Predictor


# You'll need to import Predictor and Game from ratingskit.base. Use from __future__ import annotations at the top.
# The body is short — maybe 4 lines. You loop over the games, compute the squared difference between prediction and outcome 
# for each, sum them, divide by the count.
# A few things to think about as you write:

# What if games is empty? Dividing by zero would crash. What should the function do in that case? Two reasonable options: raise a ValueError, or return 0.0 / nan. Pick one and document why.
# What does each Game give you? Look at the Game dataclass in base.py. You need three things from each game: the home name, the away name, and the outcome. You'll pass home and away to predictor.predict() to get the probability.

# Write brier_score and paste it back. Don't worry about tests yet — get the function right first, then we test it.

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

# ratingskit

A unified, composable Python library for sports and competitive rating systems.

## Why

I have been doing my own sports simulations, predictions and rankings and wanted to learn more about the science behind it and I've always wanted to build and maintain a python library.

## Status

Pre-alpha. APIs will change without notice until 0.1.0.

## Install

```bash
pip install ratingskit
```

## Quick start

```python
from ratingskit import Elo

elo = Elo(k=20, initial=1500, home_advantage=55)
elo.update("Patriots", "Bills", outcome=1.0)
prob = elo.predict("Patriots", "Bills")
```

Batch fit on a DataFrame of historical games:

```python
import pandas as pd
from ratingskit import Elo

games = pd.read_csv("nfl_games.csv")  # columns: home, away, outcome, date
elo = Elo()
elo.fit(games)
```

## Design principles

- **One API across systems.** `update`, `predict`, `rating`, `ratings`, `fit` mean the same thing whether you use Elo, Glicko-2, Massey, or Colley.
- **Composability over inheritance.** Modifiers like time decay and margin-of-victory adjustments are wrappers, not subclasses.
- **Calibration as a first-class citizen.** Built-in reliability diagrams, Brier scores, and recalibration.
- **Sport-agnostic.** Use it for NFL, NBA, chess, esports, or your CPU-vs-CPU sim league.

## Roadmap

- [x] Elo with margin-of-victory and home-advantage hooks
- [x] Glicko-2
- [ ] Massey
- [ ] Colley
- [ ] TrueSkill (v0.2)
- [ ] Bradley-Terry (v0.2)
- [ ] Calibration utilities (Brier, log loss, reliability diagrams, Platt/isotonic)
- [ ] Backtesting harness with walk-forward evaluation
- [ ] Parameter tuning (grid search, Optuna integration)
- [ ] Composable modifiers (TimeDecay, MarginAdjusted, HomeAdvantage)

## Development

Requires [uv](https://github.com/astral-sh/uv).

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

## License

MIT

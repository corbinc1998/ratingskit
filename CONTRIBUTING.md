# Contributing

## Setup

```bash
uv sync --all-extras
uv run pre-commit install
```

## Workflow

```bash
uv run ruff check .         # lint
uv run ruff format .        # format
uv run mypy                 # type check
uv run pytest               # test
uv run pytest --cov         # with coverage
```

## Adding a rating system

1. Create `src/ratingskit/systems/<name>.py` with a class that subclasses
   `RatingSystem` from `ratingskit.base`.
2. Export it from `src/ratingskit/__init__.py`.
3. Add tests under `tests/systems/test_<name>.py`. Include at minimum:
   - A property-based test verifying `predict(a, b) + predict(b, a) == 1.0`
     (for non-draw systems).
   - A regression test against a known reference implementation or paper
     example.
   - A determinism test: identical input sequences produce identical state.
4. Update `CHANGELOG.md` under `[Unreleased]`.

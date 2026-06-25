from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ratingskit.base import RatingSystem

if TYPE_CHECKING:
    from datetime import date

_SCALE_FACTOR = 173.7178  # public R / internal mu conversion
_INTERNAL_MEAN = 1500.0
_CONVERGENCE_TOLERANCE = 1e-6
_MAX_ITERATIONS = 100


@dataclass(frozen=True, slots=True)
class Glicko2Rating:
    point: float  # public-scale rating (R)
    deviation: float  # public-scale rating deviation (RD)
    volatility: float  # sigma (no scale conversion needed)


@dataclass(frozen=True, slots=True)
class _InternalRating:
    mu: float
    phi: float
    sigma: float


class Glicko2(RatingSystem):
    def __init__(
        self,
        *,
        initial: float = 1500.0,
        initial_deviation: float = 350.0,
        initial_volatility: float = 0.06,
        tau: float = 0.5,
    ) -> None:
        self.initial: float = initial
        self.initial_deviation: float = initial_deviation
        self.initial_volatility: float = initial_volatility
        self.tau: float = tau
        self._initial_mu: float = (initial - _INTERNAL_MEAN) / _SCALE_FACTOR
        self._initial_phi: float = initial_deviation / _SCALE_FACTOR
        self._ratings: dict[str, _InternalRating] = {}

    def _get_or_create(self, name: str) -> _InternalRating:
        existing = self._ratings.get(name)
        if existing is None:
            new = _InternalRating(self._initial_mu, self._initial_phi, self.initial_volatility)
            self._ratings[name] = new
            return new
        return existing

    def rating(self, competitor: str) -> Glicko2Rating:
        internal = self._get_or_create(competitor)
        r = _SCALE_FACTOR * internal.mu + _INTERNAL_MEAN
        rd = _SCALE_FACTOR * internal.phi
        return Glicko2Rating(r, rd, internal.sigma)

    def ratings(self) -> dict[str, Glicko2Rating]:
        return {name: self.rating(name) for name in self._ratings}

    def _g(self, phi: float) -> float:
        return 1 / math.sqrt(1 + 3 * phi**2 / math.pi**2)

    def predict(self, home: str, away: str) -> float:
        r_home = self._get_or_create(home)
        r_away = self._get_or_create(away)
        g = self._g(r_away.phi)
        return 1 / (1 + math.exp(-g * (r_home.mu - r_away.mu)))

    def _update_one(
        self,
        player: _InternalRating,
        opponent: _InternalRating,
        outcome: float,
    ) -> _InternalRating:
        """Compute one player's new state after a one-game period."""
        g_opp = self._g(opponent.phi)
        e_opp = 1.0 / (1.0 + math.exp(-g_opp * (player.mu - opponent.mu)))

        v = 1.0 / (g_opp * g_opp * e_opp * (1.0 - e_opp))
        delta = v * g_opp * (outcome - e_opp)

        sigma_new = self._new_volatility(player.sigma, player.phi, v, delta)

        phi_star = math.sqrt(player.phi * player.phi + sigma_new * sigma_new)
        phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
        mu_new = player.mu + phi_new * phi_new * g_opp * (outcome - e_opp)

        return _InternalRating(mu=mu_new, phi=phi_new, sigma=sigma_new)

    def _new_volatility(
        self,
        sigma: float,
        phi: float,
        v: float,
        delta: float,
    ) -> float:
        """Update volatility via the Illinois algorithm (Glickman Section 5.5)."""
        a = math.log(sigma * sigma)
        tau_sq = self.tau * self.tau
        delta_sq = delta * delta
        phi_sq = phi * phi

        def f(x: float) -> float:
            e_x = math.exp(x)
            num = e_x * (delta_sq - phi_sq - v - e_x)
            den = 2.0 * (phi_sq + v + e_x) ** 2
            return num / den - (x - a) / tau_sq

        upper_a = a
        if delta_sq > phi_sq + v:
            upper_b = math.log(delta_sq - phi_sq - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            upper_b = a - k * self.tau

        f_a = f(upper_a)
        f_b = f(upper_b)
        for _ in range(_MAX_ITERATIONS):
            if abs(upper_b - upper_a) <= _CONVERGENCE_TOLERANCE:
                break
            upper_c = upper_a + (upper_a - upper_b) * f_a / (f_b - f_a)
            f_c = f(upper_c)
            if f_c * f_b <= 0:
                upper_a = upper_b
                f_a = f_b
            else:
                f_a = f_a / 2.0
            upper_b = upper_c
            f_b = f_c

        return math.exp(upper_a / 2.0)

    def update(
        self,
        home: str,
        away: str,
        outcome: float,
        *,
        date: date | None = None,  # noqa: ARG002
        home_score: float | None = None,  # noqa: ARG002
        away_score: float | None = None,  # noqa: ARG002
    ) -> None:
        """Apply a single game result as a one-game rating period."""
        if not 0.0 <= outcome <= 1.0:
            msg = f"outcome must be in [0.0, 1.0], got {outcome}"
            raise ValueError(msg)

        home_pre = self._get_or_create(home)
        away_pre = self._get_or_create(away)

        home_post = self._update_one(home_pre, away_pre, outcome)
        away_post = self._update_one(away_pre, home_pre, 1.0 - outcome)

        self._ratings[home] = home_post
        self._ratings[away] = away_post

"""Stake sizing and hard risk limits."""
from __future__ import annotations
from .odds import kelly_fraction


def stake(bankroll: float, prob: float, odds: float, cfg: dict, scale: float = 1.0) -> float:
    f = kelly_fraction(prob, odds) * scale * cfg["kelly_fraction"]
    f = min(f, cfg["max_stake_pct"])
    s = round(max(0.0, bankroll * f), 2)
    if cfg.get("min_stake_pct") and 0 < s < cfg["min_stake_pct"] * bankroll:
        return 0.0  # dust stake: below the publish threshold, never placed
    return s


class DailyGuard:
    """Blocks new bets once the day's realised loss exceeds the limit."""
    def __init__(self, bankroll: float, limit_pct: float):
        self.limit = bankroll * limit_pct
        self.pnl = 0.0

    def record(self, pnl: float):
        self.pnl += pnl

    def allowed(self) -> bool:
        return self.pnl > -self.limit


class ExposureGuard:
    """Caps new exposure per match and per day (correlated bets on one match count together)."""
    def __init__(self, bankroll: float, match_pct: float, day_pct: float):
        self.match_cap, self.day_cap = bankroll * match_pct, bankroll * day_pct
        self.day, self.by_match = 0.0, {}

    def clip(self, match_key, s: float) -> float:
        room = min(self.match_cap - self.by_match.get(match_key, 0.0), self.day_cap - self.day)
        s = round(max(0.0, min(s, room)), 2)
        if s > 0:
            self.by_match[match_key] = self.by_match.get(match_key, 0.0) + s
            self.day += s
        return s

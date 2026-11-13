"""Odds maths: implied probabilities, de-vigging, edge, Kelly."""
from __future__ import annotations
import numpy as np


def devig_proportional(odds) -> np.ndarray:
    p = 1.0 / np.asarray(odds, dtype=float)
    return p / p.sum()


def devig_power(odds, tol: float = 1e-10) -> np.ndarray:
    """Power method: find k so that sum(p_i^k) == 1. Handles favourite-longshot bias better."""
    p = 1.0 / np.asarray(odds, dtype=float)
    if p.sum() <= 1.0:
        return p / p.sum()
    lo, hi = 1.0, 20.0
    for _ in range(200):
        k = (lo + hi) / 2
        s = (p ** k).sum()
        if abs(s - 1) < tol:
            break
        if s > 1:
            lo = k
        else:
            hi = k
    q = p ** k
    return q / q.sum()


def edge(prob: float, odds: float) -> float:
    """Expected profit per unit staked."""
    return prob * odds - 1.0


def kelly_fraction(prob: float, odds: float) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (b * prob - (1 - prob)) / b)


def get_odds(row, cols):
    """Decimal odds from row columns, or None if any is missing/invalid."""
    if not cols:
        return None
    try:
        v = np.array([row[c] for c in cols], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    return v if np.all(np.isfinite(v)) and np.all(v > 1.0) else None


def get_probs(row, cols):
    """Probabilities (e.g. Polymarket mid prices) normalised to sum to 1, or None."""
    if not cols:
        return None
    try:
        v = np.array([row[c] for c in cols], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if not (np.all(np.isfinite(v)) and np.all(v > 0) and 0.5 < v.sum() < 1.6):
        return None
    return v / v.sum()


def haircut(odds, h: float):
    """Shave a fraction h off the PROFIT part of the odds (use to model worse local prices/margins)."""
    o = np.asarray(odds, float)
    return np.maximum(1.0, 1.0 + (o - 1.0) * (1.0 - h))

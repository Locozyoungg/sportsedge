"""Goal-based markets from a score grid: Over/Under, BTTS, Asian handicap; plus settlement for every market."""
from __future__ import annotations
import numpy as np


def ou_probs(grid: np.ndarray, line: float = 2.5):
    tot = np.add.outer(np.arange(grid.shape[0]), np.arange(grid.shape[1]))
    return float(grid[tot > line].sum()), float(grid[tot < line].sum())


def btts(grid: np.ndarray) -> float:
    return float(grid[1:, 1:].sum())


def _simple_ah(grid, h):
    """Home side at half/whole line h. Returns (win, push, loss)."""
    d = np.subtract.outer(np.arange(grid.shape[0]), np.arange(grid.shape[1])) + h
    return float(grid[d > 1e-9].sum()), float(grid[np.abs(d) <= 1e-9].sum()), float(grid[d < -1e-9].sum())


def ah_probs(grid: np.ndarray, line: float, side: str = "H"):
    """(win, push, loss) for an Asian handicap bet. line = HOME handicap. Quarter lines split the stake."""
    if side == "A":
        grid, line = grid.T, -line
    q = round(line * 4) % 2 != 0       # quarter line (x.25 / x.75)
    lines = [line - 0.25, line + 0.25] if q else [line]
    parts = np.array([_simple_ah(grid, l) for l in lines])
    w, p, l = parts.mean(axis=0)
    return float(w), float(p), float(l)


def settle_bet(market: str, selection: str, line, gh: int, ga: int) -> float:
    """Profit per unit stake as a multiple of (odds-1) for a win: returns (win_fraction, loss_fraction).
    Use pnl = stake*(odds-1)*win_frac - stake*loss_frac."""
    market = market.replace("PM:", "")
    if market == "1X2":
        res = "H" if gh > ga else "A" if ga > gh else "D"
        return (1.0, 0.0) if res == selection else (0.0, 1.0)
    if market == "OU25" or market == "OU":
        ln = 2.5 if line is None else float(line)
        t = gh + ga
        if t == ln:
            return 0.0, 0.0
        over = t > ln
        return (1.0, 0.0) if (over and selection == "O") or (not over and selection == "U") else (0.0, 1.0)
    if market == "AH":
        h = float(line)
        if selection == "A":
            gh, ga, h = ga, gh, -h
        q = round(h * 4) % 2 != 0
        lines = [h - 0.25, h + 0.25] if q else [h]
        wf = lf = 0.0
        for l in lines:
            d = gh - ga + l
            if d > 1e-9: wf += 1 / len(lines)
            elif d < -1e-9: lf += 1 / len(lines)
        return wf, lf
    raise ValueError(market)


def pnl(market, selection, line, gh, ga, stake, odds) -> float:
    wf, lf = settle_bet(market, selection, line, gh, ga)
    return stake * (odds - 1) * wf - stake * lf

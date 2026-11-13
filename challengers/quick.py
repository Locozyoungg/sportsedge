"""Example shadow challenger: quick-adapt Poisson.

Same Dixon-Coles family as the champion but with a much shorter half-life (recent form dominates)
and lighter regularisation. If team strengths genuinely shift fast (manager changes, injuries,
promoted teams), this beats the champion — the shadow gate will find out without risking a shilling.
Challenger contract: NAME, fit(hist, cfg), knows(model, row), probs(model, row) -> [H, D, A].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sportsedge.model import PoissonModel

NAME = "quick"
HALF_LIFE = 45.0
REG = 0.2


def fit(hist: pd.DataFrame, cfg: dict):
    return PoissonModel(half_life_days=HALF_LIFE, reg=REG,
                        max_goals=cfg["model"]["max_goals"], dc=True, newteam_prior=False).fit(hist)


def knows(model, row) -> bool:
    return model.knows(row["HomeTeam"], row["AwayTeam"])


def probs(model, row) -> np.ndarray:
    return model.predict_1x2(row["HomeTeam"], row["AwayTeam"])

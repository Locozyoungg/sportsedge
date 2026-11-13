"""Leak-free sequential features: Elo, form, xG-proxy from shots, rest/congestion, market and injury signals.
Contract: features(match) uses ONLY matches before it; update(match) is called after features(match)."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from .odds import devig_power, get_odds
from .external import NullProvider

NAN = float("nan")


def xg_proxy(shots, sot):
    """Crude xG proxy from shot counts (no event data in the free feed): on-target worth far more than off-target."""
    if shots is None or sot is None or not np.isfinite(shots) or not np.isfinite(sot):
        return NAN
    return 0.30 * sot + 0.04 * max(shots - sot, 0.0)


def _mean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else NAN


class FeatureState:
    def __init__(self, k=18.0, home_adv=65.0, form_n=5, shots_n=8, sharp_cols=None, bet_cols=None):
        self.k, self.ha, self.fn, self.sn = k, home_adv, form_n, shots_n
        self.sharp_cols, self.bet_cols = sharp_cols or [], bet_cols or []
        self.elo = defaultdict(lambda: 1500.0)
        self.log = defaultdict(list)

    def _team(self, team, date):
        recs = self.log[team]
        if not recs:
            return dict(pts=NAN, gd=NAN, xgf=NAN, xga=NAN, rest=NAN, n14=0)
        f, s = recs[-self.fn:], recs[-self.sn:]
        return dict(pts=_mean([r["pts"] for r in f]), gd=_mean([r["gf"] - r["ga"] for r in f]),
                    xgf=_mean([r["xgf"] for r in s]), xga=_mean([r["xga"] for r in s]),
                    rest=min(float((date - recs[-1]["date"]).days), 14.0),
                    n14=sum(1 for r in recs[-8:] if 0 <= (date - r["date"]).days <= 14))

    def features(self, home, away, date, row=None, ext=None) -> dict:
        h, a = self._team(home, date), self._team(away, date)
        f = {"elo_h": self.elo[home], "elo_a": self.elo[away],
             "elo_diff": self.elo[home] + self.ha - self.elo[away],
             "form_pts_diff": h["pts"] - a["pts"], "form_gd_diff": h["gd"] - a["gd"],
             "xg_diff": (h["xgf"] - h["xga"]) - (a["xgf"] - a["xga"]),
             "rest_h": h["rest"], "rest_a": a["rest"], "rest_diff": h["rest"] - a["rest"],
             "cong_h": float(h["n14"]), "cong_a": float(a["n14"]),
             "inj_h": ext.impact(home, date) if ext else 0.0, "inj_a": ext.impact(away, date) if ext else 0.0}
        mk = None
        if row is not None:
            so, bo = get_odds(row, self.sharp_cols), get_odds(row, self.bet_cols)
            ref = so if so is not None else bo
            if ref is not None:
                mk = devig_power(ref)
            if bo is not None:
                f["overround"] = float((1 / bo).sum() - 1)
        f.setdefault("overround", NAN)
        f.update(mkt_h=mk[0] if mk is not None else NAN, mkt_d=mk[1] if mk is not None else NAN,
                 mkt_a=mk[2] if mk is not None else NAN,
                 mkt_lr=float(np.log(mk[0] / mk[2])) if mk is not None else NAN)
        return f

    def update(self, row):
        h, a, gh, ga = row["HomeTeam"], row["AwayTeam"], float(row["FTHG"]), float(row["FTAG"])
        exp_h = 1 / (1 + 10 ** (-(self.elo[h] + self.ha - self.elo[a]) / 400))
        s = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
        shift = self.k * (np.log(abs(gh - ga) + 1) + 1) * (s - exp_h)
        self.elo[h] += shift
        self.elo[a] -= shift
        g = lambda c: float(row[c]) if c in row and np.isfinite(row.get(c, NAN)) else NAN
        xh, xa = xg_proxy(g("HS"), g("HST")), xg_proxy(g("AS"), g("AST"))
        d = row["Date"]
        self.log[h].append(dict(date=d, pts=3 if gh > ga else 1 if gh == ga else 0, gf=gh, ga=ga, xgf=xh, xga=xa))
        self.log[a].append(dict(date=d, pts=3 if ga > gh else 1 if gh == ga else 0, gf=ga, ga=gh, xgf=xa, xga=xh))

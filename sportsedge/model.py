"""Time-decayed Poisson attack/defence model (football). Fit per league."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson


class PoissonModel:
    def __init__(self, half_life_days: float = 200, reg: float = 0.5, max_goals: int = 10,
                 dc: bool = True, newteam_prior: bool = False):
        self.half_life = half_life_days
        self.reg = reg
        self.max_goals = max_goals
        self.dc = dc                      # Dixon-Coles low-score correction
        self.newteam_prior = newteam_prior  # let promoted/unknown teams be predicted from a weak-team prior
        self.rho = 0.0
        self.teams: dict[str, int] = {}
        self.params: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, as_of=None) -> "PoissonModel":
        df = df.dropna(subset=["FTHG", "FTAG"])
        as_of = pd.Timestamp(as_of) if as_of is not None else df["Date"].max() + pd.Timedelta(days=1)
        df = df[df["Date"] < as_of]
        teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
        self.teams = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        h = df.HomeTeam.map(self.teams).to_numpy()
        a = df.AwayTeam.map(self.teams).to_numpy()
        yh = df.FTHG.to_numpy(float)
        ya = df.FTAG.to_numpy(float)
        age = (as_of - df.Date).dt.days.to_numpy(float)
        w = 0.5 ** (age / self.half_life)
        reg = self.reg

        def f(x):
            mu, home = x[0], x[1]
            att, dfn = x[2:2 + n], x[2 + n:]
            eh = mu + home + att[h] - dfn[a]
            ea = mu + att[a] - dfn[h]
            lh, la = np.exp(eh), np.exp(ea)
            nll = (w * (lh - yh * eh)).sum() + (w * (la - ya * ea)).sum()
            nll += reg * ((att ** 2).sum() + (dfn ** 2).sum())
            rh, ra = w * (lh - yh), w * (la - ya)
            g = np.zeros_like(x)
            g[0] = rh.sum() + ra.sum()
            g[1] = rh.sum()
            ga = np.zeros(n); gd = np.zeros(n)
            np.add.at(ga, h, rh); np.add.at(gd, a, -rh)
            np.add.at(ga, a, ra); np.add.at(gd, h, -ra)
            g[2:2 + n] = ga + 2 * reg * att
            g[2 + n:] = gd + 2 * reg * dfn
            return nll, g

        x0 = np.zeros(2 + 2 * n)
        x0[0] = np.log(max(np.average(np.r_[yh, ya], weights=np.r_[w, w]), 0.5))
        self.params = minimize(f, x0, jac=True, method="L-BFGS-B").x
        x = self.params
        att, dfn = x[2:2 + n], x[2 + n:]
        self.prior_att, self.prior_def = float(np.percentile(att, 20)), float(np.percentile(dfn, 20))
        if self.dc:
            lh = np.exp(x[0] + x[1] + att[h] - dfn[a])
            la = np.exp(x[0] + att[a] - dfn[h])
            self.rho = self._fit_rho(yh, ya, lh, la, w)
        return self

    @staticmethod
    def _tau(yh, ya, lh, la, rho):
        t = np.ones_like(lh)
        t = np.where((yh == 0) & (ya == 0), 1 - lh * la * rho, t)
        t = np.where((yh == 0) & (ya == 1), 1 + lh * rho, t)
        t = np.where((yh == 1) & (ya == 0), 1 + la * rho, t)
        t = np.where((yh == 1) & (ya == 1), 1 - rho, t)
        return t

    def _fit_rho(self, yh, ya, lh, la, w):
        obj = lambda r: -(w * np.log(np.clip(self._tau(yh, ya, lh, la, r), 1e-6, None))).sum()
        return float(minimize_scalar(obj, bounds=(-0.3, 0.3), method="bounded").x)

    def _att_def(self, team: str):
        n = len(self.teams)
        if team in self.teams:
            i = self.teams[team]
            return self.params[2 + i], self.params[2 + n + i]
        return self.prior_att, self.prior_def

    def _lambdas(self, home: str, away: str):
        mu, hadv = self.params[0], self.params[1]
        ah, dh = self._att_def(home)
        aa, da = self._att_def(away)
        return np.exp(mu + hadv + ah - da), np.exp(mu + aa - dh)

    def knows(self, home: str, away: str) -> bool:
        if self.params is None:
            return False
        return self.newteam_prior or (home in self.teams and away in self.teams)

    def score_grid(self, home: str, away: str) -> np.ndarray:
        lh, la = self._lambdas(home, away)
        k = np.arange(self.max_goals + 1)
        g = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
        if self.dc and self.rho:
            g[0, 0] *= 1 - lh * la * self.rho
            g[0, 1] *= 1 + lh * self.rho
            g[1, 0] *= 1 + la * self.rho
            g[1, 1] *= 1 - self.rho
        return g / g.sum()

    def predict_1x2(self, home: str, away: str) -> np.ndarray:
        g = self.score_grid(home, away)
        return np.array([np.tril(g, -1).sum(), np.trace(g), np.triu(g, 1).sum()])  # H, D, A

    def predict_over25(self, home: str, away: str) -> float:
        g = self.score_grid(home, away)
        idx = np.add.outer(np.arange(g.shape[0]), np.arange(g.shape[1]))
        return float(g[idx >= 3].sum())

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        lh, la = self._lambdas(home, away)
        return float(lh), float(la)

    def top_scores(self, home: str, away: str, k: int = 3) -> list[tuple[str, float]]:
        g = self.score_grid(home, away)
        flat = [(f"{i}-{j}", float(g[i, j])) for i in range(g.shape[0]) for j in range(g.shape[1])]
        return sorted(flat, key=lambda t: -t[1])[:k]

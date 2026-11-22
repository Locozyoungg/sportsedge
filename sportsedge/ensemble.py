"""Stacked ensemble: Dixon-Coles Poisson + Elo + form/xG-proxy/rest + market + injuries -> logistic regression + gradient boosting.
Everything is walk-forward: each prediction uses only data available before kickoff."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import markets as mk
from .features import FeatureState
from .model import PoissonModel
from .external import NullProvider

BASE_FEATURES = ["elo_diff", "elo_h", "elo_a", "form_pts_diff", "form_gd_diff", "xg_diff", "rest_h", "rest_a",
                 "rest_diff", "cong_h", "cong_a", "mkt_h", "mkt_d", "mkt_a", "mkt_lr", "overround", "inj_h", "inj_a"]
POIS = ["pois_h", "pois_d", "pois_a", "pois_lr"]
FEATURES = BASE_FEATURES + POIS
IDX = {"H": 0, "D": 1, "A": 2}


def pois_feats(p) -> dict:
    return {"pois_h": p[0], "pois_d": p[1], "pois_a": p[2], "pois_lr": float(np.log(max(p[0], 1e-6) / max(p[2], 1e-6)))}


def _pmodel(cfg):
    m = cfg["model"]
    return PoissonModel(m["half_life_days"], m["reg"], m["max_goals"], m.get("dc", True),
                        m.get("newteam_prior", False), m.get("max_tail_mass", 0.20))


def _state(cfg):
    b = cfg["betting"]
    return FeatureState(sharp_cols=b["sharp_cols"], bet_cols=b["venues"]["bookmaker"]["cols"])


def poisson_oos(g: pd.DataFrame, cfg) -> dict:
    """rid -> out-of-sample Poisson 1X2 probs for one league (refit every refit_days)."""
    out, kc, L = {}, cfg["backtest"], cfg["learning"]
    g = g.sort_values("Date")
    if len(g) <= kc["min_train_matches"]:
        return out
    cur, last = g.Date.iloc[kc["min_train_matches"]], g.Date.max()
    while cur <= last:
        end = cur + pd.Timedelta(days=L["refit_days"])
        win = g[(g.Date >= cur) & (g.Date < end)]
        if len(win):
            m = _pmodel(cfg).fit(g, as_of=cur)
            for rid, r in win.iterrows():
                if m.knows(r.HomeTeam, r.AwayTeam):
                    out[rid] = m.predict_1x2(r.HomeTeam, r.AwayTeam)
        cur = end
    return out


def build_table(df: pd.DataFrame, cfg, ext=None):
    """Feature table indexed by rid (+ final per-league FeatureState for live prediction)."""
    ext = ext or NullProvider()
    rows, states = [], {}
    for div, g in df.groupby("Div"):
        g = g.sort_values("Date")
        st = _state(cfg)
        po = poisson_oos(g, cfg)
        for rid, r in g.iterrows():
            f = st.features(r.HomeTeam, r.AwayTeam, r.Date, r, ext)
            f.update(pois_feats(po[rid]) if rid in po else {k: np.nan for k in POIS})
            f.update(rid=rid, Div=div, Date=r.Date, y=IDX.get(r.get("FTR"), -1) if isinstance(r.get("FTR"), str) else -1)
            rows.append(f)
            st.update(r)
        states[div] = st
    t = pd.DataFrame(rows).set_index("rid").sort_index() if rows else pd.DataFrame(columns=FEATURES)
    return t, states


class Stacker:
    def __init__(self, cfg):
        self.cfg = cfg
        s = cfg.get("stack", {})
        self.w = s.get("weights", {"logreg": 0.5, "hgb": 0.5})
        self.models = {}

    def fit(self, X: pd.DataFrame, y) -> "Stacker":
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        X = X[FEATURES].to_numpy(float)
        seed = int(self.cfg.get("backtest", {}).get("seed", 0))
        self.models = {
            "logreg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=500)),
            "hgb": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=120, l2_regularization=1.0,
                                                  min_samples_leaf=40, random_state=seed),
        }
        for m in self.models.values():
            m.fit(X, np.asarray(y))
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, float).reshape(-1, len(FEATURES))
        total, wsum = 0.0, 0.0
        for name, m in self.models.items():
            p = np.zeros((len(X), 3))
            p[:, m.classes_.astype(int)] = m.predict_proba(X)
            total = total + self.w.get(name, 0.0) * p
            wsum += self.w.get(name, 0.0)
        q = np.clip(total / wsum, 1e-6, 1)
        return q / q.sum(axis=1, keepdims=True)


def stack_walk_forward(table: pd.DataFrame, cfg) -> pd.DataFrame:
    """Pooled across leagues, chronological. Returns DataFrame(rid, p_h, p_d, p_a)."""
    t = table.dropna(subset=["pois_h"]).sort_values("Date")
    t = t[t.y >= 0]
    n0, L = cfg.get("stack", {}).get("min_train", 300), cfg["learning"]
    if len(t) <= n0:
        return pd.DataFrame(columns=["p_h", "p_d", "p_a"])
    cur, last = t.Date.iloc[n0], t.Date.max()
    out = []
    while cur <= last:
        end = cur + pd.Timedelta(days=L["refit_days"])
        win, train = t[(t.Date >= cur) & (t.Date < end)], t[t.Date < cur]
        if len(win) and len(train) >= n0:
            P = Stacker(cfg).fit(train, train.y).predict_proba(win[FEATURES].to_numpy(float))
            out.append(pd.DataFrame(P, index=win.index, columns=["p_h", "p_d", "p_a"]))
        cur = end
    return pd.concat(out).sort_index() if out else pd.DataFrame(columns=["p_h", "p_d", "p_a"])


class PoissonPredictor:
    kind = "poisson"

    def __init__(self, model, ou_b: float = 0.0, ext=None, injury_strength: float = 0.15, use_injuries: bool = False):
        self.model = model
        self.ou_b = ou_b
        self.ext = ext                # InjuryProvider (B5); NullProvider when no injuries data exists
        self.injury_k = float(injury_strength)
        self.use_injuries = bool(use_injuries)

    def _scale(self, row) -> tuple[float, float]:
        """Attack-strength multipliers for key absences (capped, config-controlled). 1.0 when off/absent."""
        if not self.use_injuries or self.ext is None:
            return (1.0, 1.0)
        ih = float(self.ext.impact(row["HomeTeam"], row["Date"]))
        ia = float(self.ext.impact(row["AwayTeam"], row["Date"]))
        return (float(np.exp(-self.injury_k * min(ih, 5.0))), float(np.exp(-self.injury_k * min(ia, 5.0))))

    def knows(self, row):
        return self.model.knows(row["HomeTeam"], row["AwayTeam"])

    def one_x_two(self, row):
        g = self.grid(row)
        return np.array([np.tril(g, -1).sum(), np.trace(g), np.triu(g, 1).sum()])

    def grid(self, row):
        return self.model.score_grid(row["HomeTeam"], row["AwayTeam"], self._scale(row))

    def xg(self, row):
        lh, la = self.model.expected_goals(row["HomeTeam"], row["AwayTeam"])
        sh, sa = self._scale(row)
        return float(lh * sh), float(la * sa)

    def ou_prob(self, row):
        """P(over 2.5) with the leak-free per-league totals-bias correction applied (AUDIT item 4)."""
        from . import markets as mk
        p = mk.ou_probs(self.grid(row), 2.5)[0]
        return mk.apply_ou_bias(p, self.ou_b)


class StackPredictor(PoissonPredictor):
    kind = "stack"

    def __init__(self, model, stacker, feat_fn, ou_b: float = 0.0, ext=None,
                 injury_strength: float = 0.15, use_injuries: bool = False):
        super().__init__(model, ou_b, ext, injury_strength, use_injuries)
        self.stacker, self.feat_fn = stacker, feat_fn

    def one_x_two(self, row):
        f = dict(self.feat_fn(row))
        f.update(pois_feats(self.model.predict_1x2(row["HomeTeam"], row["AwayTeam"])))
        return self.stacker.predict_proba([f.get(k, np.nan) for k in FEATURES])[0]


def make_predictors(hist: pd.DataFrame, cfg, ext=None) -> dict:
    """Per-league predictors fit on ALL history (live use). type=stack builds the table once and trains the stacker."""
    preds = {}
    use_stack = cfg["model"].get("type") == "stack"
    use_inj = cfg["model"].get("use_injuries", False)
    inj_k = cfg["model"].get("injury_strength", 0.15)
    if use_stack:
        table, states = build_table(hist, cfg, ext)
        tr = table.dropna(subset=["pois_h"])
        tr = tr[tr.y >= 0]
        stacker = Stacker(cfg).fit(tr, tr.y) if len(tr) >= cfg.get("stack", {}).get("min_train", 300) else None
    for div, g in hist.groupby("Div"):
        if len(g) < 200:
            continue
        pm = _pmodel(cfg).fit(g)
        ou_b = mk.fit_ou_bias(pm, g)
        if use_stack and stacker is not None:
            st = states[div]
            preds[div] = StackPredictor(pm, stacker, lambda r, st=st: st.features(r["HomeTeam"], r["AwayTeam"], r["Date"], r, ext), ou_b, ext, inj_k, use_inj)
        else:
            preds[div] = PoissonPredictor(pm, ou_b, ext, inj_k, use_inj)
    return preds

import numpy as np, pandas as pd, yaml, pathlib
from sportsedge.odds import devig_power, devig_proportional, kelly_fraction, edge
from sportsedge.model import PoissonModel
from sportsedge.backtest import run_backtest

CFG = yaml.safe_load(open(pathlib.Path(__file__).parents[1] / "config.yaml"))


def synth(seasons=3, n_teams=16, seed=1):
    rng = np.random.default_rng(seed)
    att, dfn = rng.normal(0, .25, n_teams), rng.normal(0, .25, n_teams)
    rows, d0 = [], pd.Timestamp("2022-08-01")
    for s in range(seasons):
        k = 0
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j: continue
                lh, la = np.exp(.25 + .3 + att[i] - dfn[j]), np.exp(.25 + att[j] - dfn[i])
                g = np.arange(11)
                from scipy.stats import poisson
                M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la)); M /= M.sum()
                p = np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])
                hg, ag = rng.poisson(lh), rng.poisson(la)
                def odds(m, noise): 
                    q = np.clip(p * np.exp(rng.normal(0, noise, 3)), 1e-3, None); q /= q.sum()
                    return 1 / (q * (1 + m))
                b, ps = odds(.06, .05), odds(.025, .01)
                sh = lambda g_: max(0, int(g_ * 5 + rng.integers(0, 6)))
                hst, ast = max(0, int(hg * 1.8 + rng.integers(0, 3))), max(0, int(ag * 1.8 + rng.integers(0, 3)))
                over = M[np.add.outer(g, g) > 2.5].sum()
                def two(m, p1, noise):
                    q = np.clip(p1 * np.exp(rng.normal(0, noise)), .02, .98)
                    return 1 / (q * (1 + m / 2)), 1 / ((1 - q) * (1 + m / 2))
                bo, bu = two(.06, over, .05); po, pu = two(.025, over, .01)
                rows.append(dict(Div="T", Date=d0 + pd.Timedelta(days=int(s * 365 + k * 0.9)), HomeTeam=f"T{i}", AwayTeam=f"T{j}",
                                 FTHG=hg, FTAG=ag, FTR="H" if hg > ag else "A" if ag > hg else "D",
                                 B365H=b[0], B365D=b[1], B365A=b[2], PSH=ps[0], PSD=ps[1], PSA=ps[2],
                                 PSCH=ps[0], PSCD=ps[1], PSCA=ps[2],
                                 HS=sh(lh), AS=sh(la), HST=hst, AST=ast,
                                 **{"B365>2.5": bo, "B365<2.5": bu, "P>2.5": po, "P<2.5": pu}))
                k += 1
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def test_devig_sums_to_one():
    for f in (devig_power, devig_proportional):
        assert abs(f([2.1, 3.4, 3.8]).sum() - 1) < 1e-9


def test_kelly_and_edge():
    assert kelly_fraction(0.4, 2.0) == 0.0
    assert kelly_fraction(0.6, 2.0) > 0
    assert abs(edge(0.5, 2.2) - 0.1) < 1e-9


def test_model_recovers_probabilities():
    df = synth()
    m = PoissonModel().fit(df.iloc[:-50])
    p = m.predict_1x2("T1", "T2")
    assert abs(p.sum() - 1) < 1e-9 and (p > 0).all()


def test_backtest_no_lookahead_and_runs():
    df = synth(seasons=2)
    res = run_backtest(df, CFG)
    assert res["n_predictions"] > 100
    assert res["logloss_model"] < 1.2

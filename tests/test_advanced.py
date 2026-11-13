import copy, numpy as np, pandas as pd, yaml, pathlib, pytest
from scipy.stats import poisson
from sportsedge import markets as mk, names
from sportsedge.model import PoissonModel
from sportsedge.features import FeatureState
from sportsedge.ensemble import build_table, stack_walk_forward, FEATURES, PoissonPredictor
from sportsedge.strategy import probabilities, candidates, best_bets
from sportsedge.backtest import run_backtest
from sportsedge.risk import ExposureGuard
from sportsedge.validate import validate_history
from tests.test_core import synth

CFG = yaml.safe_load(open(pathlib.Path(__file__).parents[1] / "config.yaml"))
CFG["backtest"]["min_train_matches"] = 150
CFG["stack"]["min_train"] = 120


def grid(lh=1.6, la=1.1):
    k = np.arange(11)
    g = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
    return g / g.sum()


def test_dixon_coles_grid_normalised_and_fitted():
    df = synth(seasons=3)
    m = PoissonModel(dc=True).fit(df)
    g = m.score_grid("T1", "T2")
    assert abs(g.sum() - 1) < 1e-9 and -0.3 <= m.rho <= 0.3


def test_new_team_prior():
    df = synth(seasons=2)
    m = PoissonModel(newteam_prior=True).fit(df)
    assert m.knows("T1", "BRAND_NEW") and abs(m.predict_1x2("T1", "BRAND_NEW").sum() - 1) < 1e-9
    assert not PoissonModel().fit(df).knows("T1", "BRAND_NEW")


def test_ou_btts_and_ah():
    g = grid()
    o, u = mk.ou_probs(g, 2.5)
    assert abs(o + u - 1) < 1e-9
    w, p, l = mk.ah_probs(g, -0.5)
    assert p == 0 and abs(w + l - 1) < 1e-9
    w2, p2, l2 = mk.ah_probs(g, 0.0, "A")          # draw-no-bet away
    assert p2 > 0 and abs(w2 + p2 + l2 - 1) < 1e-9
    assert mk.settle_bet("AH", "H", -0.25, 1, 1) == (0.0, 0.5)   # quarter line: half loss
    assert mk.settle_bet("AH", "H", -0.75, 2, 1) == (0.5, 0.0)   # half win
    assert mk.pnl("OU25", "O", None, 2, 1, 10, 2.0) == 10
    assert mk.pnl("1X2", "D", None, 1, 1, 10, 3.0) == 20


def test_names():
    assert names.match_team("Manchester United", ["Man United", "Man City"]) == "Man United"
    assert names.match_team("Nottingham Forest", ["Nott'm Forest"]) == "Nott'm Forest"
    assert names.match_team("Paris Saint Germain", ["Paris SG", "Lyon"]) == "Paris SG"
    assert names.match_team("Zzyzx", ["Man United"]) is None


def test_features_do_not_leak_future():
    df = synth(seasons=1).reset_index(drop=True)
    st = FeatureState()
    for i, r in df.iloc[:60].iterrows():
        st.update(r)
    r = df.iloc[60]
    before = st.features(r.HomeTeam, r.AwayTeam, r.Date, r)
    st.update(r)                                          # the match itself is only added AFTER features
    assert st.features(r.HomeTeam, r.AwayTeam, r.Date, r) != before


def test_table_and_stack_walk_forward_run():
    df = synth(seasons=3)
    cfg = copy.deepcopy(CFG)
    t, states = build_table(df, cfg)
    assert set(FEATURES) <= set(t.columns) and "T" in states
    ps = stack_walk_forward(t, cfg)
    assert len(ps) > 100 and np.allclose(ps.sum(axis=1), 1)
    # stacked rows must come strictly after the training burn-in
    assert t.loc[ps.index, "Date"].min() > t.sort_values("Date").Date.iloc[100]


def test_multi_market_candidates_and_exposure():
    df = synth(seasons=3)
    cfg = copy.deepcopy(CFG)
    cfg["betting"]["min_edge"] = -1.0                     # force candidates to exist
    cfg["betting"]["max_bets_per_match"] = 5
    m = PoissonModel().fit(df.iloc[:-20])
    pred = PoissonPredictor(m)
    row = df.iloc[-1]
    pr = probabilities(pred, row, cfg)
    assert pr and "ou" in pr
    markets = {c["market"] for c in candidates(pr, cfg)}
    assert "1X2" in markets and "OU25" in markets
    g = ExposureGuard(1000, 0.03, 0.08)
    assert g.clip(("a", "b"), 25) == 25 and g.clip(("a", "b"), 25) == 5   # per-match cap 30


def test_polymarket_venue_candidate():
    df = synth(seasons=3)
    cfg = copy.deepcopy(CFG)
    m = PoissonModel().fit(df.iloc[:-20])
    row = df.iloc[-1].copy()
    row["PM_H"], row["PM_D"], row["PM_A"] = 9.0, 9.0, 9.0        # absurdly generous Polymarket prices
    row["PM_MID_H"] = 0.4
    cfg["betting"]["max_odds"] = 20
    pr = probabilities(PoissonPredictor(m), row, cfg)
    assert "polymarket" in pr["x2"]["venues"]
    assert any(c["venue"] == "polymarket" and c["market"] == "PM:1X2" for c in candidates(pr, cfg))


def test_backtest_multi_market_and_stack():
    df = synth(seasons=3)
    cfg = copy.deepcopy(CFG)
    r = run_backtest(df, cfg)
    assert r["n_predictions"] > 100
    cfg["model"]["type"] = "stack"
    r2 = run_backtest(df, cfg)
    assert r2["n_predictions"] > 100 and r2["logloss_model"] < 1.2


def test_validate_catches_problems():
    df = synth(seasons=1)
    assert validate_history(df, CFG)["ok"]
    bad = pd.concat([df, df.iloc[:3]])
    assert not validate_history(bad, CFG)["ok"]
    bad2 = df.copy(); i0 = bad2.index[0]; bad2.loc[i0, "FTHG"] = bad2.loc[i0, "FTAG"] + 3; bad2.loc[i0, "FTR"] = "A"
    assert not validate_history(bad2, CFG)["ok"]

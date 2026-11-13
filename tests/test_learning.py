import copy, numpy as np, pandas as pd, yaml, pathlib
from sportsedge.calibration import Calibrator
from sportsedge import learning, postmortem
from sportsedge.ledger import Ledger
from tests.test_core import synth

CFG = yaml.safe_load(open(pathlib.Path(__file__).parents[1] / "config.yaml"))


def cfg_tmp(tmp_path):
    c = copy.deepcopy(CFG)
    c["learning"]["models_dir"] = str(tmp_path / "models")
    c["learning"]["reports_dir"] = str(tmp_path / "reports")
    c["learning"]["min_holdout_matches"] = 100
    c["learning"]["refit_days"] = 28
    c["learning"]["grid"] = {"half_life_days": [200], "reg": [0.2, 0.5], "market_weight": [0.0, 0.5, 1.0]}
    c["backtest"]["min_train_matches"] = 150
    c["stack"]["min_train"] = 120
    return learning.apply_champion(c)


def test_calibrator_fixes_overconfidence():
    rng = np.random.default_rng(0)
    true = rng.dirichlet([4, 3, 4], 4000)
    y = np.array([rng.choice(3, p=p) for p in true])
    over = Calibrator(T=1.6).apply(true)            # an overconfident model
    fixed = Calibrator.fit(over, y).apply(over)
    ll = lambda P: -np.log(P[np.arange(len(y)), y]).mean()
    assert ll(fixed) < ll(over) - 0.01


def test_improve_promotes_misspecified_champion(tmp_path):
    c = cfg_tmp(tmp_path)
    c["model"].update(reg=300.0, market_weight=0.0)   # deliberately terrible: team strengths crushed to zero
    d = learning.improve(c, synth(seasons=4))
    assert d["decision"] == "promoted", d
    assert learning.load_champion(c)["version"] == 1
    # cooldown: a second run right away must NOT promote again
    c2 = learning.apply_champion(copy.deepcopy(c))
    assert learning.improve(c2, synth(seasons=4))["decision"] == "kept"


def test_improve_keeps_good_champion(tmp_path):
    c = cfg_tmp(tmp_path)
    c["model"].update(reg=0.5, half_life_days=200, market_weight=0.5)
    d = learning.improve(c, synth(seasons=4))
    assert d["decision"] == "kept"


def test_rollback(tmp_path):
    c = cfg_tmp(tmp_path)
    learning.save_version(c, {"half_life_days": 200, "reg": 0.5, "market_weight": 0.5}, Calibrator(), {})
    learning.save_version(c, {"half_life_days": 120, "reg": 1.0, "market_weight": 0.2}, Calibrator(), {})
    learning.rollback(c, 1)
    assert learning.load_champion(c)["version"] == 1


def test_bet_review_flags_bad_luck_vs_error():
    ok = pd.DataFrame(dict(stake=[10] * 200, odds=[2.0] * 200, prob=[0.52] * 200, pnl=[10] * 100 + [-10] * 100))
    assert "variance" in postmortem.bet_review(ok)["verdict"]
    bad = pd.DataFrame(dict(stake=[10] * 200, odds=[2.0] * 200, prob=[0.6] * 200, pnl=[10] * 60 + [-10] * 140))
    assert "WORSE" in postmortem.bet_review(bad)["verdict"]


def test_prediction_log_is_write_once(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    row = {"Date": pd.Timestamp("2026-11-14"), "HomeTeam": "A", "AwayTeam": "B"}
    pr = {"model_raw": [.4, .3, .3], "market": [.4, .3, .3], "final": [.4, .3, .3], "bet_odds": [2.4, 3.3, 3.3]}
    led.log_prediction("E0", row, pr)
    pr2 = dict(pr, final=[.9, .05, .05])
    led.log_prediction("E0", row, pr2)               # a later "improved" prediction must not overwrite history
    assert led.db.execute("SELECT COUNT(*), MAX(pf_h) FROM predictions").fetchone() == (1, 0.4)

"""Phase 3 tests: weekly metrics, segmentation, safe auto-adjustment (reductions/pauses only), rollback."""
import copy
import json
import os
import pathlib

import numpy as np
import pandas as pd
import yaml

from sportsedge import eval as ev
from sportsedge.ledger import Ledger

CFG = yaml.safe_load(open(pathlib.Path(__file__).parents[1] / "config.yaml"))


def cfg_tmp(tmp_path, monkeypatch):
    c = copy.deepcopy(CFG)
    c["learning"]["models_dir"] = str(tmp_path / "models")
    c["learning"]["reports_dir"] = str(tmp_path / "reports")
    c["paths"] = {"ledger": str(tmp_path / "l.sqlite"), "raw_dir": str(tmp_path / "raw")}
    (tmp_path / "raw").mkdir()
    monkeypatch.chdir(tmp_path)
    return c


def _bet_df(rows):
    """Minimal settled-bets DataFrame shaped like Ledger.settled_bets()."""
    base = dict(ts="2026-11-01T00:00:00", match_date="2026-11-01", home="A", away="B",
                market="1X2", selection="H", odds=2.0, stake=10.0, prob=0.5, edge=0.05,
                status="won", pnl=10.0, mode="paper", line=None, venue="bookmaker", ref=None,
                div="E0", clv=0.01, message_id=None, odds_ts=None, book=None, p_market=0.48,
                closing_odds=0.5)
    out = []
    for r in rows:
        d = dict(base)
        d.update({k: v for k, v in r.items() if v is not None})
        out.append(d)
    return pd.DataFrame(out)


def test_overall_metrics_are_exact():
    bets = _bet_df([
        dict(pnl=10.0, prob=0.5, status="won", clv=0.02),
        dict(pnl=-10.0, prob=0.7, status="lost", clv=-0.01),
        dict(pnl=8.0, prob=0.3, status="won", clv=0.03),
        dict(pnl=-12.0, prob=0.6, status="lost", clv=0.01),
    ])
    o = ev.overall(bets, 1000)
    assert o["n"] == 4 and abs(o["roi"] - (-4 / 40)) < 1e-9
    assert abs(o["brier"] - 0.3975) < 1e-9
    assert abs(o["ll"] - 1.00435) < 1e-3
    assert abs(o["clv"] - 0.0125) < 1e-9
    assert abs(o["max_drawdown"] - ((996 - 1010) / 1010)) < 1e-9
    assert o["calibration"]


def test_segmentation_dims_and_weekly_file(tmp_path, monkeypatch):
    c = cfg_tmp(tmp_path, monkeypatch)
    led = Ledger(c["paths"]["ledger"])
    rng = np.random.default_rng(0)
    for i in range(24):
        d = dict(date="2026-11-01", home="A", away="B", market=["1X2", "OU25", "AH"][i % 3],
                 selection="H", odds=1.4 + (i % 5) * 0.4, stake=5, prob=0.5, edge=0.02 + (i % 4) * 0.03,
                 div=["E0", "I1"][i % 2], clv=rng.normal(0, 0.01), p_market=0.5)
        led.add(d)
    for bid in range(1, 25):
        led.settle_pnl(bid, 6.0 if bid % 2 else -5.0, clv=0.01, closing_odds=0.5)
    seg = ev.segmented(led.settled_bets())
    assert len(seg) >= 3 and set(seg.columns) >= {"segment", "n", "roi", "clv", "brier", "ll", "bias"}
    md = ev.weekly_report(c, led)
    assert "Weekly report" in md and "Calibration" in md
    assert os.path.exists(c["learning"]["reports_dir"] + "/weekly_" + __import__("datetime").date.today().isoformat() + ".md")


def _fill(led, n, div, market, odds, prob, clv_mean, win_rate, rng, start=1):
    """Place and settle n bets in one segment."""
    for i in range(n):
        d = dict(date="2026-11-01", home=f"{div}-{i}", away=f"{div}-away{i}", market=market, selection="H", odds=odds,
                 stake=5, prob=prob, edge=prob * odds - 1, div=div, p_market=0.5)
        led.add(d)
        won = rng.random() < win_rate
        led.settle_pnl(start + i, 5 * (odds - 1) if won else -5, clv=clv_mean, closing_odds=1 / odds)


def test_auto_adjust_reduces_pauses_never_increases(tmp_path, monkeypatch):
    c = cfg_tmp(tmp_path, monkeypatch)
    led = Ledger(c["paths"]["ledger"])
    rng = np.random.default_rng(0)
    _fill(led, 120, "E0", "1X2", 2.0, 0.52, -0.03, 0.5, rng)        # bad CLV -> reduce
    _fill(led, 120, "I1", "1X2", 2.0, 0.52, 0.03, 0.5, rng, start=121)  # good CLV -> unchanged
    _fill(led, 320, "E0", "OU25", 1.9, 0.55, 0.0, 0.47, rng, start=241)  # bias 0.08 -> pause
    adj = ev.auto_adjust(c, led)
    st = ev._load_state()
    bad = st["segments"]["E0|1X2|1.6-2.5|2-5%"]
    good = st["segments"]["I1|1X2|1.6-2.5|2-5%"]
    paused = st["segments"]["E0|OU25|1.6-2.5|2-5%"]
    assert bad["multiplier"] == 0.5 and good["multiplier"] == 1.0 and paused["status"] == "paused"
    # never increase: an explicit lower multiplier stays where it is
    st["segments"]["I1|1X2|1.6-2.5|2-5%"]["multiplier"] = 0.25
    ev._save_state(st, c)
    ev.auto_adjust(c, led)
    assert ev._load_state()["segments"]["I1|1X2|1.6-2.5|2-5%"]["multiplier"] == 0.25
    # reversible: roll back to the first snapshot
    snaps = sorted(os.listdir(c["learning"]["models_dir"] + "/segments_history"))
    assert len(snaps) >= 2
    ev.rollback(c, snaps[0].replace(".json", ""))
    assert ev._load_state()["segments"]["I1|1X2|1.6-2.5|2-5%"]["multiplier"] == 1.0
    log_lines = [json.loads(l) for l in open(c["learning"]["models_dir"] + "/segments_log.jsonl", encoding="utf-8")]
    assert log_lines and {"action", "before", "after"} <= set(log_lines[0])


def test_apply_segment_pause_multiplier_and_recalibration(tmp_path, monkeypatch):
    c = cfg_tmp(tmp_path, monkeypatch)
    st = {"segments": {
        "E0|1X2|1.6-2.5|2-5%": {"multiplier": 0.5, "status": "active", "n": 300, "cal_kind": None, "cal": None},
        "E0|OU25|1.6-2.5|2-5%": {"multiplier": 1.0, "status": "paused", "n": 300, "cal_kind": None, "cal": None},
        "E0|1X2|2.5-4|2-5%": {"multiplier": 1.0, "status": "active", "n": 300, "cal_kind": "isotonic",
                              "cal": [0.15, 0.25, 0.35, 0.45, 0.6, 0.7, 0.78, 0.85, 0.9]},
    }, "updated": None}
    ev._save_state(st, c)
    b1 = dict(market="1X2", odds=2.0, prob=0.55, edge=0.03, scale=1.0, stake=10.0)
    ev.apply_segment(b1, c, "E0")
    assert b1["stake"] == 5.0 and b1["stake_multiplier"] == 0.5 and b1["prob"] == 0.55  # multiplier only
    b2 = dict(market="OU25", odds=2.0, prob=0.55, edge=0.03, scale=1.0, stake=10.0)
    ev.apply_segment(b2, c, "E0")
    assert b2["stake"] == 0.0 and b2["paused_segment"]
    b3 = dict(market="1X2", odds=3.0, prob=0.55, edge=0.03, scale=1.0, stake=10.0)
    ev.apply_segment(b3, c, "E0")
    assert b3["prob"] > 0.55 and b3["edge"] != 0.03            # isotonic recalibration applied, edge recomputed
    b4 = dict(market="1X2", odds=1.2, prob=0.55, edge=0.03, scale=1.0, stake=10.0)
    ev.apply_segment(b4, c, "E0")                             # segment without registry entry: untouched
    assert b4["stake"] == 10.0

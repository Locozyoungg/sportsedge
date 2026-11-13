"""Tests for the Phase 4 service layer: ledger CLV/message ids, publish + withdraw flow,
shadow challenger research, drift detection, dashboard helpers."""
import copy
import json

import numpy as np
import pandas as pd
import yaml
import pathlib

from sportsedge import cli
from sportsedge import postmortem
from sportsedge.challengers import run_shadow, settle_shadow, discover
from sportsedge.dashboard import drawdown_series, clv_summary, per_league_roi
from sportsedge.ledger import Ledger
from sportsedge.telegram import TelegramPublisher
from tests.test_core import synth

CFG = yaml.safe_load(open(pathlib.Path(__file__).parents[1] / "config.yaml"))


def cfg_tmp(tmp_path):
    c = copy.deepcopy(CFG)
    c["paths"] = {"raw_dir": str(tmp_path / "raw"), "ledger": str(tmp_path / "l.sqlite")}
    c["learning"]["models_dir"] = str(tmp_path / "models")
    c["learning"]["reports_dir"] = str(tmp_path / "reports")
    c["backtest"]["min_train_matches"] = 150
    c["stack"]["min_train"] = 120
    (tmp_path / "raw").mkdir()
    return c


# ---------- ledger: clv, message ids, withdrawal ----------
def test_ledger_clv_and_message_id(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    bet = dict(date="2026-11-14", home="A", away="B", market="1X2", selection="H",
               odds=2.0, stake=5, prob=0.5, edge=0.0, div="E0")
    assert led.add(bet)
    bid = led.find_bet(bet)
    assert bid is not None and led.add(bet) is False          # idempotent
    led.set_message_id(bid, 123)
    led.settle_pnl(bid, 4.9, clv=0.03)
    row = led.db.execute("SELECT message_id, clv, div, status FROM bets").fetchone()
    assert row == (123, 0.03, "E0", "won")
    assert led.unposted_bets("2026-11-01") == []
    led2 = Ledger(str(tmp_path / "l2.sqlite"))                 # older ledger without new columns migrates
    assert led2.add(dict(bet, date="2026-11-15")) is True


def test_withdraw_marks_and_excludes_from_open(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    bet = dict(date="2026-11-14", home="A", away="B", market="1X2", selection="H",
               odds=2.0, stake=5, prob=0.5, edge=0.0, div="E0")
    led.add(bet)
    bid = led.find_bet(bet)
    led.set_message_id(bid, 7)
    assert len(led.posted_open_bets()) == 1
    led.withdraw(bid)
    assert len(led.posted_open_bets()) == 0
    assert len(led.open_bets()) == 0
    n, staked, pnl = led.report()
    assert n == 1 and pnl == 0                                # withdrawn: no stake, no pnl


# ---------- publish flow ----------
class FakeTG:
    """Stands in for TelegramPublisher in the cli module: returns real-looking message ids, records traffic."""
    def __init__(self):
        self.posted, self.edited, self.sent = [], [], []

    def post(self, text):
        self.posted.append(text)
        return [1000 + len(self.posted) + len(self.sent)]

    def send(self, text):
        self.sent.append(text)
        return True

    def edit(self, mid, text):
        self.edited.append((mid, text))
        return True


FX_COLS = ["Div", "Date", "HomeTeam", "AwayTeam"] + ["B365H", "B365D", "B365A", "PSH", "PSD", "PSA",
                                                      "PSCH", "PSCD", "PSCA", "B365>2.5", "B365<2.5",
                                                      "P>2.5", "P<2.5"]


def _setup_pipeline(tmp_path, cfg, n_fx=20, shift_days=1200):
    # shift_days must put fixture dates strictly in the future relative to TODAY,
    # because the publish sweep only posts bets whose match_date >= today.
    df = synth(seasons=2)
    df.to_csv(tmp_path / "raw" / "s_T.csv", index=False)
    fx = df.tail(n_fx)[FX_COLS].copy()
    fx["Date"] = fx["Date"] + pd.Timedelta(days=shift_days)
    fx.to_csv(tmp_path / "raw" / "fixtures.csv", index=False)
    cfg["betting"]["min_edge"] = -1.0
    cfg["betting"]["max_bets_per_match"] = 1
    return df, fx


def _no_env(monkeypatch):
    for v in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHANNEL_ID", "ODDS_API_KEY"):
        monkeypatch.delenv(v, raising=False)


def test_publish_posts_once_and_sweeps_unposted(tmp_path, monkeypatch, capsys):
    c = cfg_tmp(tmp_path)
    df, _ = _setup_pipeline(tmp_path, c)
    fake = FakeTG()
    monkeypatch.setattr(cli, "TelegramPublisher", lambda: fake)
    _no_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    cli.cmd_predict(c, record=True, publish=True)
    led = Ledger(c["paths"]["ledger"])
    posted = led.db.execute("SELECT COUNT(*) FROM bets WHERE message_id IS NOT NULL").fetchone()[0]
    n = led.db.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
    assert n > 0 and posted == n                            # every placed bet got a message id
    picks = [t for t in fake.posted if "Pick:" in t]
    assert len(picks) == n
    cli.cmd_predict(c, record=True, publish=True)           # rerun: no new bets, no double-post
    n2 = led.db.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
    assert n2 == n
    picks2 = [t for t in fake.posted if "Pick:" in t]
    assert len(picks2) == n                                 # no pick was posted twice
    out = capsys.readouterr().out
    assert "nothing new" in out.lower() or "recorded 0" in out.lower()


def test_unposted_sweep_posts_bets_missing_message_ids(tmp_path, monkeypatch):
    c = cfg_tmp(tmp_path)
    df, _ = _setup_pipeline(tmp_path, c)
    _no_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cli.cmd_predict(c, record=True, publish=True)           # dry-run telegram: bets recorded, no ids stored
    led = Ledger(c["paths"]["ledger"])
    n = led.db.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
    assert n > 0 and led.db.execute("SELECT COUNT(*) FROM bets WHERE message_id IS NOT NULL").fetchone()[0] == 0
    fake = FakeTG()
    monkeypatch.setattr(cli, "TelegramPublisher", lambda: fake)
    cli.cmd_predict(c, record=True, publish=True)           # now with real creds: sweep posts the open picks
    ids = led.db.execute("SELECT COUNT(*) FROM bets WHERE message_id IS NOT NULL").fetchone()[0]
    assert ids == n


# ---------- withdraw command ----------
def test_withdraw_drops_picks_whose_edge_gone(tmp_path, monkeypatch):
    c = cfg_tmp(tmp_path)
    df, fx = _setup_pipeline(tmp_path, c)
    fake = FakeTG()
    monkeypatch.setattr(cli, "TelegramPublisher", lambda: fake)
    _no_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cli.cmd_predict(c, record=True, publish=True)
    led = Ledger(c["paths"]["ledger"])
    n = led.db.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
    assert n > 0
    # prices collapse everywhere (1X2 and totals): no edge survives anywhere
    for col in c["betting"]["venues"]["bookmaker"]["cols"] + ["B365>2.5", "B365<2.5"]:
        fx[col] = 1.01
    fx.to_csv(tmp_path / "raw" / "fixtures.csv", index=False)
    cli.cmd_withdraw(c)
    withdrawn = led.db.execute("SELECT COUNT(*) FROM bets WHERE status='withdrawn'").fetchone()[0]
    assert withdrawn == n
    assert fake.edited and all("ODDS MOVED" in t and "withdrawn" in t.lower() for _, t in fake.edited)
    led.settle_predictions(df)                              # settle must not touch withdrawn bets
    assert led.db.execute("SELECT COUNT(*) FROM bets WHERE status='withdrawn'").fetchone()[0] == n


# ---------- shadow challenger research ----------
def test_shadow_log_is_write_once_and_settles(tmp_path, monkeypatch):
    c = cfg_tmp(tmp_path)
    df = synth(seasons=2)
    df.to_csv(tmp_path / "raw" / "s_T.csv", index=False)
    train, fx = df.iloc[:-80], df.tail(80)                 # 80 > the 50-per-challenger report threshold
    fx[FX_COLS[:4]].to_csv(tmp_path / "raw" / "fixtures.csv", index=False)
    c["learning"]["shadow_min_matches"] = 10 ** 6          # huge: proposal must stay False
    r1 = run_shadow(c, train, fx)
    assert r1["logged"] > 0 and len(discover()) >= 1
    r2 = run_shadow(c, train, fx)                           # write-once: second run logs nothing new
    assert r2["logged"] == 0
    led = Ledger(c["paths"]["ledger"])
    for _, row in fx.iterrows():                            # champion predictions for the same matches
        led.log_prediction(row["Div"], row, {"model_raw": [.4, .3, .3], "market": [.4, .3, .3],
                                             "final": [.4, .3, .3], "bet_odds": [2.4, 3.3, 3.3]})
    led.settle_predictions(df)
    rep = settle_shadow(c, df, led)
    assert rep["n"] > 0 and rep["challengers"]
    rec = rep["challengers"][0]
    assert "ll_challenger" in rec and "ll_champion_final" in rec and rec["propose_promotion"] is False


def test_shadow_proposes_when_clearly_better(tmp_path, monkeypatch):
    c = cfg_tmp(tmp_path)
    df = synth(seasons=3)
    df.to_csv(tmp_path / "raw" / "s_T.csv", index=False)
    train, fx = df.iloc[:-200], df.tail(200)
    fx[FX_COLS[:4]].to_csv(tmp_path / "raw" / "fixtures.csv", index=False)
    c["learning"]["shadow_min_matches"] = 150
    run_shadow(c, train, fx)
    led = Ledger(c["paths"]["ledger"])
    for _, row in fx.iterrows():                            # champion "final" is garbage on purpose
        led.log_prediction(row["Div"], row, {"model_raw": [.9, .05, .05], "market": [.4, .3, .3],
                                             "final": [.9, .05, .05], "bet_odds": [1.1, 20, 20]})
    led.settle_predictions(df)
    rep = settle_shadow(c, df, led)
    assert any(r["propose_promotion"] for r in rep["challengers"]), json.dumps(rep, indent=2)
    from sportsedge.challengers import propose_text
    assert "PROPOSED" in propose_text(rep)


# ---------- drift detection ----------
def _ledger_with_predictions(c, df, pf, pk=(0.45, 0.25, 0.30)):
    led = Ledger(c["paths"]["ledger"])
    for _, row in df.iterrows():
        led.log_prediction(row["Div"], row, {"model_raw": list(pf), "market": list(pk),
                                             "final": list(pf), "bet_odds": [2.4, 3.3, 3.3]})
    led.settle_predictions(df)
    return led


def test_health_flags_league_drift_and_recovers(tmp_path):
    c = cfg_tmp(tmp_path)
    df = synth(seasons=1).iloc[:150]
    led = _ledger_with_predictions(c, df, pf=(0.1, 0.8, 0.1))  # persistently wrong vs a sane market
    reasons = postmortem.health(c, led)
    assert any("drift" in r or "T:" in r for r in reasons), reasons
    led2 = Ledger(str(tmp_path / "l2.sqlite"))                 # fresh ledger: write-once predictions
    for _, row in df.iterrows():
        led2.log_prediction(row["Div"], row, {"model_raw": [.45, .25, .30], "market": [.45, .25, .30],
                                              "final": [.45, .25, .30], "bet_odds": [2.4, 3.3, 3.3]})
    led2.settle_predictions(df)
    assert postmortem.health(c, led2) == []


# ---------- dashboard helpers ----------
def test_dashboard_helpers():
    bets = pd.DataFrame({"ts": ["2026-11-01", "2026-11-02", "2026-11-03"], "pnl": [10.0, -15.0, 20.0],
                         "match_date": ["2026-11-01"] * 3, "stake": [5.0] * 3, "clv": [0.02, 0.03, 0.01],
                         "div": ["E0", "E0", "E0"], "home": "A", "away": "B", "selection": "H",
                         "odds": 2.0, "prob": 0.5, "edge": 0.0, "status": "won", "mode": "paper"})
    dd = drawdown_series(bets, 1000)
    assert dd.bankroll.iloc[-1] == 1015 and dd.drawdown.min() < 0
    assert abs(clv_summary(bets).mean_clv[0] - 0.02) < 1e-9
    assert per_league_roi(bets).iloc[0].roi == 15 / 15


def test_fixtures_csv_bom_and_lowercase_div(tmp_path):
    """football-data fixtures.csv ships 'ï»¿Div' (latin-1-decoded UTF-8 BOM) and can use lowercase 'div'."""
    from sportsedge import data
    raw = tmp_path / "raw"
    raw.mkdir()
    for header in ("﻿Div", "ï»¿Div", "div"):  # real BOM, mojibake BOM, lowercase
        (raw / "fixtures.csv").write_text(
            f"{header},Date,HomeTeam,AwayTeam\nE0,2026-11-22,Arsenal,Chelsea\n", encoding="utf-8")
        fx = data.load_fixtures(str(raw))
        assert "Div" in fx.columns and fx.Div.iloc[0] == "E0", (header, list(fx.columns))


def test_load_history_unifies_bom_mojibake_headers(tmp_path):
    """AUDIT finding 0: seasons fetched with mojibake 'ï»¿Div' headers must not silently lose their Div.
    Regression: they used to concat under a second column and vanish from per-league training."""
    from sportsedge import data
    from sportsedge.validate import validate_history
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "2324_D1.csv").write_text(  # clean header
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nD1,2024-08-23,A,B,1,0,H\nD1,2024-08-24,C,D,2,2,D\n",
        encoding="utf-8")
    (raw / "2425_D1.csv").write_text(  # mojibake header, as fetched pre-fix; distinct matches
        "ï»¿Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nD1,2025-08-23,E,F,0,3,A\nD1,2025-08-24,G,H,1,1,D\n",
        encoding="utf-8")
    df = data.load_history(str(raw))
    assert len(df) == 4 and df.Div.notna().all() and set(df.Div) == {"D1"}
    cfg = {"betting": {"venues": {"bookmaker": {"cols": ["B365H", "B365D", "B365A"]}}}}
    assert validate_history(df, cfg)["ok"]
    # and the validator must flag NaN Div if it ever reappears
    bad = df.copy()
    bad.loc[0, "Div"] = None
    assert not validate_history(bad, cfg)["ok"]


def test_min_stake_threshold_drops_dust_bets(tmp_path, monkeypatch):
    """Phase 2 item 1: a pick whose stake is below min_stake_pct must never be placed or published."""
    from sportsedge.risk import stake
    c = cfg_tmp(tmp_path)
    b = c["betting"]
    b["min_stake_pct"] = 0.005
    assert stake(1000, 0.501, 2.0, b) == 0.0          # tiny edge -> dust stake -> 0
    assert stake(1000, 0.6, 2.0, b) > 0.0             # real edge survives
    del b["min_stake_pct"]
    assert stake(1000, 0.501, 2.0, b) > 0.0           # without the threshold, dust was possible
    # end-to-end: with the threshold, a marginal pick is filtered before the ledger
    df, fx = _setup_pipeline(tmp_path, c)
    c["betting"]["min_stake_pct"] = 1.0               # impossibly high: nothing may be placed
    fake = FakeTG()
    monkeypatch.setattr(cli, "TelegramPublisher", lambda: fake)
    _no_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cli.cmd_predict(c, record=True, publish=True)
    led = Ledger(c["paths"]["ledger"])
    assert led.db.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 0


def test_telegram_post_returns_ids_and_edit_dry_run(monkeypatch, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHANNEL_ID", raising=False)
    tg = TelegramPublisher()
    assert tg.post("short message") == [-1]
    assert tg.edit(5, "edited") is True
    assert "DRY RUN" in capsys.readouterr().out


def test_telegram_post_extracts_message_id_from_result_envelope(monkeypatch):
    """sendMessage returns the id under result.message_id — post() must find it there
    (regression: the first version read r['message_id'] and treated a successful send as failure)."""
    import sportsedge.telegram as tmod
    monkeypatch.setattr(tmod.time, "sleep", lambda s: None)
    ids = iter([42])
    tg = TelegramPublisher(token="t", chat_id="c", dry_run=False)
    monkeypatch.setattr(tg, "_call", lambda method, payload: {"ok": True, "result": {"message_id": next(ids)}})
    assert tg.post("one message") == [42]
    ids2 = iter([43])
    tg2 = TelegramPublisher(token="t", chat_id="c", dry_run=False)
    monkeypatch.setattr(tg2, "_call", lambda method, payload: {"ok": True, "result": {"message_id": next(ids2)}})
    assert tg2.send("another") is True

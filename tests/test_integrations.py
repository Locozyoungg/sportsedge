import copy, json, os, pandas as pd, numpy as np, yaml, pathlib, pytest
from sportsedge import polymarket as pmk, oddsfeed, cli, markets as mk
from sportsedge.ledger import Ledger
from sportsedge.report import format_bet
from tests.test_core import synth

CFG = yaml.safe_load(open(pathlib.Path(__file__).parents[1] / "config.yaml"))


class Resp:
    def __init__(self, data, ok=True): self.data, self.ok = data, ok
    def json(self): return self.data
    def raise_for_status(self):
        if not self.ok: raise RuntimeError("http error")


class FakeSession:
    def __init__(self, events, books, geo=None): self.events, self.books, self.geo = events, books, geo
    def get(self, url, params=None, timeout=0):
        if url.endswith("/events"): return Resp(self.events if params["offset"] == 0 else [])
        if url.endswith("/book"): return Resp(self.books[params["token_id"]])
        if "geoblock" in url: return Resp(self.geo, ok=self.geo is not None)
        raise AssertionError(url)


def pm_events():
    mkt = lambda q, tok: {"question": q, "outcomes": '["Yes","No"]', "outcomePrices": '["0.45","0.55"]',
                          "clobTokenIds": json.dumps([tok, tok + "n"]), "liquidity": "5000", "gameStartTime": "2026-11-22T15:00:00Z"}
    return [{"title": "Arsenal vs. Chelsea", "markets": [
        mkt("Will Arsenal FC win on 2026-11-22?", "tH"), mkt("Will Arsenal vs. Chelsea end in a draw?", "tD"),
        mkt("Will Chelsea win on 2026-11-22?", "tA")]}]


BOOKS = {t: {"bids": [{"price": str(b), "size": "100"}], "asks": [{"price": str(a), "size": "100"}]}
         for t, (b, a) in {"tH": (.44, .46), "tD": (.26, .28), "tA": (.27, .29)}.items()}


def test_polymarket_classify_and_scan():
    cfg = copy.deepcopy(CFG)
    fx = pd.DataFrame([{"Div": "E0", "Date": pd.Timestamp("2026-11-22"), "HomeTeam": "Arsenal", "AwayTeam": "Chelsea"}])
    client = pmk.PolymarketClient(cfg, FakeSession(pm_events(), BOOKS))
    out = pmk.scan(fx, client.events(), client, cfg)
    assert abs(out.PM_H[0] - 1 / .46) < 1e-9 and abs(out.PM_D[0] - 1 / .28) < 1e-9 and abs(out.PM_A[0] - 1 / .29) < 1e-9
    assert out.PM_TOKEN_H[0] == "tH" and abs(out.PM_MID_H[0] - .45) < 1e-9


def test_polymarket_skips_wide_spread_and_wrong_date():
    cfg = copy.deepcopy(CFG)
    wide = {t: {"bids": [{"price": "0.10", "size": "9"}], "asks": [{"price": "0.40", "size": "9"}]} for t in BOOKS}
    fx = pd.DataFrame([{"Div": "E0", "Date": pd.Timestamp("2026-11-22"), "HomeTeam": "Arsenal", "AwayTeam": "Chelsea"}])
    c = pmk.PolymarketClient(cfg, FakeSession(pm_events(), wide))
    assert pmk.scan(fx, c.events(), c, cfg)[["PM_H", "PM_D", "PM_A"]].isna().all().all()
    fx2 = fx.assign(Date=pd.Timestamp("2026-12-30"))
    c2 = pmk.PolymarketClient(cfg, FakeSession(pm_events(), BOOKS))
    assert pmk.scan(fx2, c2.events(), c2, cfg)[["PM_H"]].isna().all().all()


def test_geoblock_fails_closed():
    cfg = copy.deepcopy(CFG)
    assert pmk.check_geoblock(cfg, FakeSession([], {}, geo=None))[0] is False                 # error -> not allowed
    assert pmk.check_geoblock(cfg, FakeSession([], {}, geo={"blocked": True, "country": "XX"}))[0] is False
    assert pmk.check_geoblock(cfg, FakeSession([], {}, geo={"blocked": False}))[0] is True


def test_live_broker_gates(tmp_path, monkeypatch):
    cfg = copy.deepcopy(CFG)
    led = Ledger(str(tmp_path / "l.sqlite"))
    with pytest.raises(RuntimeError):                          # disabled in config
        pmk.PolymarketLiveBroker(cfg, led)
    cfg["polymarket"]["live"]["enabled"] = True
    monkeypatch.delenv("LIVE_CONFIRM", raising=False)
    with pytest.raises(RuntimeError):                          # no confirm env
        pmk.PolymarketLiveBroker(cfg, led)
    monkeypatch.setenv("LIVE_CONFIRM", "I_ACCEPT_RISK")
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0x1")
    with pytest.raises(RuntimeError):                          # geoblock says blocked
        pmk.PolymarketLiveBroker(cfg, led, FakeSession([], {}, geo={"blocked": True}))
    b = pmk.PolymarketLiveBroker(cfg, led, FakeSession([], {}, geo={"blocked": False}))
    (tmp_path / "STOP").write_text("x"); monkeypatch.chdir(tmp_path)
    assert b.place({"stake": 5, "odds": 2.0, "ref": "t", "home": "A", "away": "B", "selection": "H"}) is False   # kill switch


def test_odds_feed_parse_and_name_mapping():
    cfg = copy.deepcopy(CFG)
    ev = [{"home_team": "Manchester United", "away_team": "Tottenham Hotspur", "commence_time": "2026-11-22T15:00:00Z",
           "bookmakers": [
               {"key": "bookA", "markets": [{"key": "h2h", "outcomes": [{"name": "Manchester United", "price": 2.1}, {"name": "Draw", "price": 3.4}, {"name": "Tottenham Hotspur", "price": 3.6}]},
                                            {"key": "totals", "outcomes": [{"name": "Over", "price": 1.9, "point": 2.5}, {"name": "Under", "price": 1.95, "point": 2.5}]}]},
               {"key": "bookB", "markets": [{"key": "h2h", "outcomes": [{"name": "Manchester United", "price": 2.2}, {"name": "Draw", "price": 3.3}, {"name": "Tottenham Hotspur", "price": 3.5}]}]},
               {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [{"name": "Manchester United", "price": 2.15}, {"name": "Draw", "price": 3.45}, {"name": "Tottenham Hotspur", "price": 3.7}]}]}]}]
    fx = oddsfeed.parse_events(ev, "E0", cfg)
    assert fx.loc[0, "B365H"] == 2.2 and fx.loc[0, "B365D"] == 3.4 and fx.loc[0, "PSH"] == 2.15   # best-of-books + Pinnacle ref
    assert fx.loc[0, "BookH"] == "bookB" and fx.loc[0, "BookD"] == "bookA"   # Phase 2 item 8: best book tracked per selection
    hist = pd.DataFrame({"Div": "E0", "HomeTeam": ["Man United", "Tottenham"], "AwayTeam": ["Tottenham", "Man United"]})
    mapped = oddsfeed.map_names(fx, hist)
    assert mapped.loc[0, "HomeTeam"] == "Man United" and mapped.loc[0, "AwayTeam"] == "Tottenham"


def test_settle_handles_asian_handicap_and_ou(tmp_path):
    df = synth(seasons=1); raw = tmp_path / "raw"; raw.mkdir()
    df.to_csv(raw / "s_T.csv", index=False)
    cfg = copy.deepcopy(CFG); cfg["paths"] = {"raw_dir": str(raw), "ledger": str(tmp_path / "l.sqlite")}
    r = df.iloc[10]
    led = Ledger(cfg["paths"]["ledger"])
    led.add({"date": r.Date, "home": r.HomeTeam, "away": r.AwayTeam, "market": "AH", "selection": "H", "line": -0.25,
             "odds": 1.95, "stake": 10, "prob": .5, "edge": .05})
    led.add({"date": r.Date, "home": r.HomeTeam, "away": r.AwayTeam, "market": "OU25", "selection": "O", "line": 2.5,
             "odds": 1.9, "stake": 10, "prob": .5, "edge": .05})
    cli.cmd_settle(cfg)
    got = sorted(x[0] for x in led.db.execute("SELECT pnl FROM bets").fetchall())
    exp = sorted([mk.pnl("AH", "H", -0.25, int(r.FTHG), int(r.FTAG), 10, 1.95), mk.pnl("OU25", "O", 2.5, int(r.FTHG), int(r.FTAG), 10, 1.9)])
    assert np.allclose(got, exp)


def test_report_labels_markets():
    b = dict(home="A", away="B", date="2026-11-22", div="E0", odds=1.9, prob=.55, edge=.05, stake_pct=1.0, selection="O", market="OU25", line=2.5)
    assert "Over 2.5" in format_bet(b)
    b2 = dict(b, selection="A", market="AH", line=-0.5)
    assert "+0.5" in format_bet(b2)
    assert "Polymarket" in format_bet(dict(b, selection="H", market="PM:1X2", venue="polymarket"))

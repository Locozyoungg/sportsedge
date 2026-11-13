"""Live odds feed (The Odds API). Gives fresh, multi-bookmaker prices instead of stale CSV snapshots.
Needs ODDS_API_KEY. Kenyan bookmakers are not covered: use `odds_haircut` and/or data/external/local_odds.csv."""
from __future__ import annotations
import os
import pandas as pd
import requests
from . import names

BASE = "https://api.the-odds-api.com/v4"
SPORT_KEYS = {"E0": "soccer_epl", "E1": "soccer_efl_champ", "SP1": "soccer_spain_la_liga",
              "D1": "soccer_germany_bundesliga", "I1": "soccer_italy_serie_a", "F1": "soccer_france_ligue_one"}


def parse_events(events: list, div: str, cfg: dict, book_filter=None) -> pd.DataFrame:
    b = cfg["betting"]
    cols, sharp = b["venues"]["bookmaker"]["cols"], b["sharp_cols"]
    ou, rows = b["markets"]["ou"], []
    for e in events:
        best, pin, bo, po = [0.0] * 3, [None] * 3, [0.0, 0.0], [None, None]
        for bk in e.get("bookmakers", []):
            if book_filter and bk["key"] not in book_filter and bk["key"] != "pinnacle":
                continue
            for m in bk.get("markets", []):
                if m["key"] == "h2h":
                    px = {o["name"]: o["price"] for o in m["outcomes"]}
                    v = [px.get(e["home_team"]), px.get("Draw"), px.get(e["away_team"])]
                    if None in v:
                        continue
                    if bk["key"] == "pinnacle":
                        pin = v
                    elif not book_filter or bk["key"] in book_filter:
                        best = [max(a, c) for a, c in zip(best, v)]
                elif m["key"] == "totals":
                    px = {o["name"]: o["price"] for o in m["outcomes"] if o.get("point") == 2.5}
                    if "Over" in px and "Under" in px:
                        if bk["key"] == "pinnacle":
                            po = [px["Over"], px["Under"]]
                        else:
                            bo = [max(bo[0], px["Over"]), max(bo[1], px["Under"])]
        if min(best) <= 1:
            continue
        r = {"Div": div, "Date": pd.Timestamp(e["commence_time"]).tz_localize(None) if pd.Timestamp(e["commence_time"]).tzinfo is None
             else pd.Timestamp(e["commence_time"]).tz_convert(None),
             "HomeTeam": e["home_team"], "AwayTeam": e["away_team"]}
        r.update(dict(zip(cols, best)))
        if None not in pin:
            r.update(dict(zip(sharp, pin)))
        if min(bo) > 1:
            r[ou["over"]], r[ou["under"]] = bo
        if None not in po:
            r[ou["sharp_over"]], r[ou["sharp_under"]] = po
        rows.append(r)
    return pd.DataFrame(rows)


def map_names(fx: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Rename feed team names to the names used in history; drop fixtures that cannot be mapped."""
    keep = []
    for i, r in fx.iterrows():
        teams = sorted(set(hist[hist.Div == r.Div].HomeTeam) | set(hist[hist.Div == r.Div].AwayTeam))
        h, a = names.match_team(r.HomeTeam, teams), names.match_team(r.AwayTeam, teams)
        if h and a:
            fx.at[i, "HomeTeam"], fx.at[i, "AwayTeam"] = h, a
            keep.append(i)
        else:
            print("unmapped fixture:", r.HomeTeam, "v", r.AwayTeam)
    return fx.loc[keep].reset_index(drop=True)


def fetch_live_fixtures(cfg: dict, hist: pd.DataFrame, regions="eu,uk", book_filter=None) -> pd.DataFrame:
    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise SystemExit("Set ODDS_API_KEY (the-odds-api.com) to use live odds")
    frames = []
    for div in cfg["leagues"]:
        sk = SPORT_KEYS.get(div)
        if not sk:
            continue
        r = requests.get(f"{BASE}/sports/{sk}/odds", params={"apiKey": key, "regions": regions,
                         "markets": "h2h,totals", "oddsFormat": "decimal"}, timeout=30)
        if not r.ok:
            print("odds feed error", div, r.status_code)
            continue
        df = parse_events(r.json(), div, cfg, book_filter)
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return map_names(pd.concat(frames, ignore_index=True), hist)


def apply_local_odds(fx: pd.DataFrame, path: str, cfg: dict) -> pd.DataFrame:
    """Override bookmaker odds with prices YOU logged from your own (e.g. Kenyan) bookmaker.
    CSV columns: Date,HomeTeam,AwayTeam,H,D,A   (names as in history)"""
    if not os.path.exists(path):
        return fx
    loc = pd.read_csv(path, parse_dates=["Date"])
    cols = cfg["betting"]["venues"]["bookmaker"]["cols"]
    for _, r in loc.iterrows():
        m = (fx.HomeTeam == r.HomeTeam) & (fx.AwayTeam == r.AwayTeam) & (fx.Date.dt.date == r.Date.date())
        for c, k in zip(cols, ("H", "D", "A")):
            fx.loc[m, c] = r[k]
    return fx

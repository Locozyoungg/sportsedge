"""Free historical results + odds from football-data.co.uk."""
from __future__ import annotations
import io, os
import pandas as pd
import requests

BASE = "https://www.football-data.co.uk"


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip a BOM (or its latin-1 mojibake 'ï»¿') that some feeds put before the first header."""
    df.columns = [str(c).lstrip("﻿ï»¿").strip() for c in df.columns]
    return df


def _read(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(content.decode("latin-1")), on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    df = _clean_cols(df)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["Date"])


def fetch(leagues, seasons, raw_dir) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    for s in seasons:
        for lg in leagues:
            url = f"{BASE}/mmz4281/{s}/{lg}.csv"
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                _read(r.content).to_csv(f"{raw_dir}/{s}_{lg}.csv", index=False)
                print("ok  ", url)
            except Exception as e:  # keep going; some league/season combos don't exist
                print("skip", url, e)
    try:
        r = requests.get(f"{BASE}/fixtures.csv", timeout=30)
        r.raise_for_status()
        _read(r.content).to_csv(f"{raw_dir}/fixtures.csv", index=False)
        print("ok   fixtures.csv")
    except Exception as e:
        print("skip fixtures.csv", e)


def load_history(raw_dir) -> pd.DataFrame:
    frames = []
    for f in sorted(os.listdir(raw_dir)):
        if f.endswith(".csv") and f != "fixtures.csv":
            frames.append(pd.read_csv(f"{raw_dir}/{f}", parse_dates=["Date"]))
    if not frames:
        raise SystemExit("No data. Run: python -m sportsedge fetch")
    df = pd.concat(frames, ignore_index=True)
    return df.dropna(subset=["FTHG", "FTAG", "FTR"]).sort_values("Date").reset_index(drop=True)


def fixtures_path(raw_dir) -> str:
    """Prefer the live-odds file (odds feed / Polymarket overlay) if it exists and is not older than fixtures.csv."""
    base, live = f"{raw_dir}/fixtures.csv", f"{raw_dir}/fixtures_live.csv"
    if os.path.exists(live) and (not os.path.exists(base) or os.path.getmtime(live) >= os.path.getmtime(base)):
        return live
    return base


def load_fixtures(raw_dir, cfg=None) -> pd.DataFrame:
    p = fixtures_path(raw_dir)
    if not os.path.exists(p):
        raise SystemExit("No fixtures. Run: python -m sportsedge fetch")
    fx = pd.read_csv(p, parse_dates=["Date"])
    fx = _clean_cols(fx)
    if "div" in fx.columns and "Div" not in fx.columns:  # football-data fixtures.csv uses lowercase div
        fx = fx.rename(columns={"div": "Div"})
    if cfg is not None:
        from .oddsfeed import apply_local_odds
        fx = apply_local_odds(fx, "data/external/local_odds.csv", cfg)
    return fx

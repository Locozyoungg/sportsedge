"""Free historical results + odds from football-data.co.uk."""
from __future__ import annotations
import io, os
import pandas as pd
import requests

BASE = "https://www.football-data.co.uk"

CANONICAL = {"div": "Div", "date": "Date", "hometeam": "HomeTeam", "awayteam": "AwayTeam",
             "fthg": "FTHG", "ftag": "FTAG", "ftr": "FTR"}
REQUIRED = ["Div", "Date", "HomeTeam", "AwayTeam"]


def load_team_aliases(path) -> dict:
    """name -> canonical from data/team_aliases.csv (missing or empty file = no mappings).
    One club spelled two ways across seasons would otherwise train as two clubs."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, comment="#")
    if not len(df):
        return {}
    if "name" not in df.columns or "canonical" not in df.columns:
        raise ValueError(f"{path} must have columns 'name,canonical'")
    return {str(k).strip(): str(v).strip() for k, v in zip(df["name"], df["canonical"])}


def canonical_team(name, aliases: dict) -> str:
    if not aliases or pd.isna(name):
        return name
    return aliases.get(str(name), str(name))


def apply_team_aliases(df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    if not aliases:
        return df
    out = df.copy()
    for col in ("HomeTeam", "AwayTeam"):
        if col in out.columns:
            out[col] = out[col].map(lambda t: canonical_team(t, aliases))
    return out


def _decode(content: bytes) -> str:
    """utf-8-sig first (covers real UTF-8 BOMs), latin-1 fallback (covers pre-fix mojibake files)."""
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse dd/mm/yyyy and dd/mm/yy (dayfirst). Fail loudly on anything unparseable;
    blank cells become NaT (rows with no date are dropped by the caller)."""
    s = series.astype(str).str.strip()
    empty = (s.str.len() == 0) | s.str.lower().isin(["nan", "nat", "none"])
    out = pd.to_datetime(s.where(~empty), format="%Y-%m-%d", errors="coerce")
    still = out.isna() & ~empty
    if still.any():  # football-data ships dd/mm/yyyy
        out2 = pd.to_datetime(s[still], format="%d/%m/%Y", errors="coerce")
        out.loc[still] = out2
        still = out.isna() & ~empty
    if still.any():  # ... and some feeds use dd/mm/yy
        out2 = pd.to_datetime(s[still], format="%d/%m/%y", errors="coerce")
        out.loc[still] = out2
        still = out.isna() & ~empty
    if still.any():  # live-odds fixtures carry full timestamps
        out2 = pd.to_datetime(s[still], errors="coerce")
        out.loc[still] = out2
        still = out.isna() & ~empty
    if still.any():
        rows = list(series.index[still][:10])
        raise ValueError(f"unparseable dates at rows {rows}: {list(series[still][:5])} "
                         f"(expected dd/mm/yyyy or dd/mm/yy)")
    return out


def _normalize_frame(df: pd.DataFrame, aliases: dict | None = None) -> pd.DataFrame:
    """Single shared cleaning path for EVERY csv we load (history, fixtures, feeds):
    1. strip BOM / latin-1 mojibake 'ï»¿' / whitespace from headers, canonicalise lowercase names;
    2. parse dates robustly (dd/mm/yyyy or dd/mm/yy), raising loudly on garbage;
    3. canonicalise team names via data/team_aliases.csv;
    4. de-duplicate on (Div, Date, HomeTeam, AwayTeam)."""
    df.columns = [str(c).lstrip("﻿ï»¿").strip() for c in df.columns]
    df = df.rename(columns={c: CANONICAL[c.lower()] for c in df.columns if c.lower() in CANONICAL})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"file missing required columns {missing} (headers: {list(df.columns)})")
    if "Date" in df.columns:
        df["Date"] = _parse_dates(df["Date"])
        df = df.dropna(subset=["Date"]).reset_index(drop=True)
    if aliases:
        df = apply_team_aliases(df, aliases)
    n = len(df)
    df = df.drop_duplicates(REQUIRED, keep="first").reset_index(drop=True)
    if len(df) != n:
        print(f"dropped {n - len(df)} duplicate rows on (Div, Date, HomeTeam, AwayTeam)")
    return df


def _read_csv(source: str | bytes) -> pd.DataFrame:
    content = source if isinstance(source, str) else _decode(source)
    return pd.read_csv(io.StringIO(content), on_bad_lines="skip")


def _read(content: bytes) -> pd.DataFrame:
    df = _read_csv(content)
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    return _normalize_frame(df)


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


def _aliases_for(raw_dir) -> dict:
    """Team-name aliases live next to the raw data: data/team_aliases.csv."""
    return load_team_aliases(os.path.join(os.path.dirname(raw_dir), "team_aliases.csv"))


def load_history(raw_dir) -> pd.DataFrame:
    frames, colsets = [], {}
    aliases = _aliases_for(raw_dir)
    for f in sorted(os.listdir(raw_dir)):
        if not f.endswith(".csv") or f in ("fixtures.csv", "fixtures_live.csv"):
            continue
        with open(f"{raw_dir}/{f}", "rb") as fh:
            df = _normalize_frame(_read_csv(_decode(fh.read())), aliases)
        colsets[f] = set(df.columns)
        missing_scores = [c for c in ("FTHG", "FTAG", "FTR") if c not in df.columns]
        if missing_scores:
            raise SystemExit(f"{f} is missing score columns {missing_scores} — refusing to silently drop its rows")
        frames.append(df)
    if not frames:
        raise SystemExit("No data. Run: python -m sportsedge fetch")
    sets = [set(c) for c in colsets.values()]
    allcols = set().union(*sets)
    diffs = {f: sorted(allcols - colsets[f]) for f in colsets if allcols - colsets[f]}
    if diffs:
        missing = sorted(set().union(*diffs.values()))
        print(f"warning: files have differing column sets (odds columns vary by season); "
              f"missing in some files: {missing[:25]}{' ...' if len(missing) > 25 else ''}")
    df = pd.concat(frames, ignore_index=True)
    n = len(df)
    df = df.drop_duplicates(REQUIRED, keep="first")
    if len(df) != n:
        print(f"dropped {n - len(df)} duplicate rows across files on (Div, Date, HomeTeam, AwayTeam)")
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
    with open(p, "rb") as fh:
        fx = _normalize_frame(_read_csv(_decode(fh.read())), _aliases_for(raw_dir))
    if cfg is not None:
        from .oddsfeed import apply_local_odds
        fx = apply_local_odds(fx, "data/external/local_odds.csv", cfg)
    return fx

"""Data quality gate. Fatal issues should pause autopilot (bad data silently ruins models)."""
from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def _season_label(dates: pd.Series) -> pd.Series:
    """Season string '2425' for a date: seasons run Aug..Jul. NaN dates label as 'unknown'
    (already reported as fatal by the missing-Date check)."""
    start = dates.dt.year.where(dates.dt.month >= 8, dates.dt.year - 1)
    return start.map(lambda y: f"{int(y) % 100:02d}{(int(y) + 1) % 100:02d}" if pd.notna(y) else "unknown")


def validate_history(df: pd.DataFrame, cfg) -> dict:
    fatal, warn = [], []
    v = cfg.get("validate", {})
    min_rows = int(v.get("min_rows", 50))
    max_goals = int(v.get("max_goals", 15))
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        return {"ok": False, "fatal": [f"missing columns {miss}"], "warn": []}
    nan_div = int(df["Div"].isna().sum())
    if nan_div:
        fatal.append(f"{nan_div} rows with missing Div (BOM-mojibake headers? see AUDIT.md finding 0)")
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")  # never crash on a string Date column
    bad_dates = int(df["Date"].isna().sum())
    if bad_dates:
        fatal.append(f"{bad_dates} rows with missing Date")
    for col in ("HomeTeam", "AwayTeam"):
        bad = int(df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum())
        if bad:
            fatal.append(f"{bad} rows with missing {col}")
    if df.duplicated(["Div", "Date", "HomeTeam", "AwayTeam"]).any():
        fatal.append(f"{int(df.duplicated(['Div', 'Date', 'HomeTeam', 'AwayTeam']).sum())} duplicate matches")
    if (df[["FTHG", "FTAG"]] < 0).any().any():
        fatal.append("negative goals")
    if (df[["FTHG", "FTAG"]] > max_goals).any().any():
        fatal.append(f"implausible score: more than {max_goals} goals in a match")
    res = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG < df.FTAG, "A", "D"))
    if (res != df.FTR.to_numpy()).any():
        fatal.append("FTR inconsistent with goals")
    if (df.Date > pd.Timestamp.today() + pd.Timedelta(days=1)).any():
        fatal.append("results dated in the future")
    for div, g in df.groupby("Div"):
        for season, s in g.groupby(_season_label(g.Date)):
            if len(s) < min_rows:
                fatal.append(f"{div} {season}: only {len(s)} rows (below validate.min_rows={min_rows})")
    for c in cfg["betting"]["venues"]["bookmaker"]["cols"]:
        if c in df.columns:
            bad = df[c].dropna()
            if (bad <= 1).any():
                fatal.append(f"{c}: odds <= 1")
        else:
            warn.append(f"column {c} absent")
    cols = cfg["betting"]["venues"]["bookmaker"]["cols"]
    if all(c in df.columns for c in cols):
        o = df[cols].dropna()
        ov = (1 / o).sum(axis=1)
        if len(o) and ((ov < 0.98) | (ov > 1.25)).mean() > 0.02:
            fatal.append("overround outside 0.98-1.25 for >2% of rows (check odds columns)")
        share = df[cols[0]].isna().mean()
        if share > 0.2:
            warn.append(f"{share:.0%} of rows lack bookmaker odds")
    for div, g in df.groupby("Div"):
        n = len(set(g.HomeTeam) | set(g.AwayTeam))
        if n > 40:
            warn.append(f"{div}: {n} distinct team names (name changes?)")
    return {"ok": not fatal, "fatal": fatal, "warn": warn}

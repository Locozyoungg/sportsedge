"""Data quality gate. Fatal issues should pause autopilot (bad data silently ruins models)."""
from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def validate_history(df: pd.DataFrame, cfg) -> dict:
    fatal, warn = [], []
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        return {"ok": False, "fatal": [f"missing columns {miss}"], "warn": []}
    nan_div = int(df["Div"].isna().sum())
    if nan_div:
        fatal.append(f"{nan_div} rows with missing Div (BOM-mojibake headers? see AUDIT.md finding 0)")
    if df.duplicated(["Div", "Date", "HomeTeam", "AwayTeam"]).any():
        fatal.append(f"{int(df.duplicated(['Div', 'Date', 'HomeTeam', 'AwayTeam']).sum())} duplicate matches")
    if (df[["FTHG", "FTAG"]] < 0).any().any():
        fatal.append("negative goals")
    res = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG < df.FTAG, "A", "D"))
    if (res != df.FTR.to_numpy()).any():
        fatal.append("FTR inconsistent with goals")
    if (df.Date > pd.Timestamp.today() + pd.Timedelta(days=1)).any():
        fatal.append("results dated in the future")
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
            warn.append("overround outside 0.98-1.25 for >2% of rows (check odds columns)")
        share = df[cols[0]].isna().mean()
        if share > 0.2:
            warn.append(f"{share:.0%} of rows lack bookmaker odds")
    for div, g in df.groupby("Div"):
        n = len(set(g.HomeTeam) | set(g.AwayTeam))
        if n > 40:
            warn.append(f"{div}: {n} distinct team names (name changes?)")
    return {"ok": not fatal, "fatal": fatal, "warn": warn}

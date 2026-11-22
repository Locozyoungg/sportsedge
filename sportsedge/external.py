"""Pluggable external signals (injuries/suspensions/lineups). Impact scale: 0 = nothing, ~1 = a key starter out,
2+ = several key players out. Providers must only return information that was PUBLIC at `date` (no look-ahead)."""
from __future__ import annotations
import os
import pandas as pd


class InjuryProvider:
    def impact(self, team: str, date) -> float:
        return 0.0


class NullProvider(InjuryProvider):
    pass


class CsvInjuryProvider(InjuryProvider):
    """CSV columns: date, team, impact, source  (source URL is mandatory so every number is auditable)."""
    def __init__(self, path: str, max_age_days: int = 7):
        df = pd.read_csv(path, parse_dates=["date"])
        if "source" not in df.columns or df["source"].isna().any():
            raise ValueError("every injury row needs a source")
        self.df = df.sort_values("date")
        self.max_age = pd.Timedelta(days=max_age_days)

    def impact(self, team: str, date) -> float:
        d = pd.Timestamp(date)
        r = self.df[(self.df.team == team) & (self.df.date <= d) & (self.df.date >= d - self.max_age)]
        return float(r.iloc[-1]["impact"]) if len(r) else 0.0


class ApiFootballProvider(InjuryProvider):
    """TEMPLATE for a paid/free-tier API (e.g. API-Football). Implement `impact` by mapping the injuries endpoint
    to the 0..N impact scale (weight by player minutes/rating), cache results, respect rate limits, store source+timestamp.
    Left as NotImplemented on purpose: verify the provider's current schema and terms before wiring it."""
    def __init__(self, api_key: str | None = None):
        self.key = api_key or os.getenv("API_FOOTBALL_KEY")

    def impact(self, team, date):
        raise NotImplementedError("implement against the provider's current docs")


def load_provider(cfg) -> InjuryProvider:
    p = cfg.get("external", {}).get("injuries_csv")
    return CsvInjuryProvider(p) if p and os.path.exists(p) else NullProvider()


class XgProvider:
    """Real xG source (Understat/FBref) to replace the shots-based proxy in features.xg_proxy.
    Contract: returns (xG_for, xG_against) for `team` in matches up to `date`, or None when unknown.
    Providers must only use data that was PUBLIC at `date` (no look-ahead)."""

    def team_xg(self, team: str, date) -> tuple[float, float] | None:
        raise NotImplementedError


class UnderstatProvider(XgProvider):
    """BLOCKED (2026-11-22): Understat has no official public API and its data is served via
    per-match HTML pages (scraping required, no rate-limit contract, no stable schema); FBref
    blocks automated scraping by robots.txt/ToS and needs a per-team ID mapping that changes
    between seasons. Wiring this means a scraper + a cache + legal sign-off before any model
    change — out of scope for the engine phase. The shots-based xg_proxy remains in use."""
    pass

"""Team-name normalisation so football-data, odds feeds and Polymarket titles can be joined reliably."""
from __future__ import annotations
import difflib, re, unicodedata

ALIASES = {
    "manchester united": "man united", "man utd": "man united", "manchester city": "man city",
    "nottingham forest": "nottm forest", "newcastle united": "newcastle", "newcastle utd": "newcastle",
    "tottenham hotspur": "tottenham", "spurs": "tottenham", "wolverhampton wanderers": "wolves",
    "brighton and hove albion": "brighton", "west ham united": "west ham", "leeds united": "leeds",
    "leicester city": "leicester", "atletico madrid": "ath madrid", "athletic bilbao": "ath bilbao",
    "athletic club": "ath bilbao", "real betis": "betis", "real sociedad": "sociedad",
    "borussia dortmund": "dortmund", "borussia monchengladbach": "mgladbach", "gladbach": "mgladbach",
    "bayer leverkusen": "leverkusen", "eintracht frankfurt": "ein frankfurt", "inter milan": "inter",
    "internazionale": "inter", "ac milan": "milan", "paris saint germain": "paris sg", "psg": "paris sg",
    "bayern munchen": "bayern munich", "rb leipzig": "rb leipzig", "saint etienne": "st etienne",
    "queens park rangers": "qpr", "west bromwich albion": "west brom",
    "hamburger sv": "hamburg", "hamburg sv": "hamburg",
}
STOP = {"fc", "cf", "afc", "fk", "sc", "the", "club", "calcio", "1"}


def _clean(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ").replace("'", "").replace(".", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize(name: str) -> str:
    s = _clean(name)
    s = ALIASES.get(s, s)
    toks = [t for t in s.split() if t not in STOP]
    s = " ".join(toks) if toks else s
    return ALIASES.get(s, s)


def match_team(name: str, candidates, cutoff: float = 0.82):
    """Best candidate for `name`, or None. Exact normalised match first, then token-subset, then fuzzy."""
    cands = {c: normalize(c) for c in candidates}
    n = normalize(name)
    for c, nc in cands.items():
        if nc == n:
            return c
    nt = set(n.split())
    for c, nc in cands.items():
        ct = set(nc.split())
        if ct and nt and (ct <= nt or nt <= ct) and min(len(ct), len(nt)) >= 1 and len(n) > 3 and len(nc) > 3:
            return c
    close = difflib.get_close_matches(n, list(cands.values()), n=1, cutoff=cutoff)
    if close:
        return next(c for c, nc in cands.items() if nc == close[0])
    return None

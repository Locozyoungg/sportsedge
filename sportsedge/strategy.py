"""Model + prices -> candidate bets across markets (1X2, Over/Under 2.5, Asian handicap) and venues (bookmaker, Polymarket)."""
from __future__ import annotations
import datetime as dt
import json
import os
import numpy as np
from . import markets as mk
from .odds import devig_power, edge, get_odds, get_probs, haircut  # noqa: F401  (get_odds re-exported)

OUT = ["H", "D", "A"]


def _edge_cap(cfg: dict, div) -> float:
    return float(cfg["betting"].get("edge_caps", {}).get(div, cfg["betting"].get("max_edge", 0.15)))


def _log_suspicious(cfg: dict, bet: dict, div) -> None:
    """Flag and log (never publish) edges above the league cap: model-vs-market disagreement."""
    d = cfg.get("learning", {}).get("models_dir", "models")
    os.makedirs(d, exist_ok=True)
    rec = {"ts": dt.datetime.now(dt.UTC).replace(tzinfo=None).isoformat(), "div": div, "market": bet["market"], "venue": bet["venue"],
           "selection": bet["selection"], "line": bet.get("line"), "odds": bet["odds"],
           "edge": bet["edge"], "cap": _edge_cap(cfg, div)}
    with open(os.path.join(d, "suspicious_log.jsonl"), "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _ev_flag(x):
    return x is not None and np.all(np.isfinite(x))


def probabilities(pred, row, cfg) -> dict | None:
    if not pred.knows(row):
        return None
    b = cfg["betting"]
    w = cfg["model"]["market_weight"]
    venues = {}
    for name, v in b["venues"].items():
        o = get_odds(row, v["cols"])
        if o is None:
            continue
        if name == "bookmaker":
            o = haircut(o, b.get("odds_haircut", 0.0))
        fee = v.get("fee", 0.0)
        if fee:
            o = 1.0 / ((1.0 / o) * (1 + fee))
        venues[name] = o
    if not venues:
        return None
    sharp = get_odds(row, b["sharp_cols"])
    pmmid = get_probs(row, b.get("polymarket_mid_cols", []))
    market = devig_power(sharp) if sharp is not None else pmmid if pmmid is not None else devig_power(next(iter(venues.values())))
    raw = np.asarray(pred.one_x_two(row))
    cal = cfg.get("_calibrator")
    pm = cal.apply(raw) if (cal is not None and pred.kind == "poisson") else raw
    out = {"x2": {"model_raw": raw, "model": pm, "market": market, "final": w * market + (1 - w) * pm,
                  "venues": venues, "bet_odds": venues.get("bookmaker", next(iter(venues.values())))},
           "pred": pred}
    en = b["markets"]["enabled"]
    grid = None
    if "OU25" in en:
        c = b["markets"]["ou"]
        oo = get_odds(row, [c["over"], c["under"]])
        if oo is not None:
            grid = pred.grid(row)
            po = pred.ou_prob(row) if hasattr(pred, "ou_prob") else mk.ou_probs(grid, 2.5)[0]
            so = get_odds(row, [c["sharp_over"], c["sharp_under"]])
            m = devig_power(so if so is not None else oo)[0]
            out["ou"] = {"model": po, "market": m, "final": w * m + (1 - w) * po, "odds": haircut(oo, b.get("odds_haircut", 0.0))}
    if "AH" in en:
        c = b["markets"]["ah"]
        try:
            line = float(row[c["line"]])
        except (KeyError, TypeError, ValueError):
            line = np.nan
        oa = get_odds(row, [c["home"], c["away"]])
        if oa is not None and np.isfinite(line):
            grid = grid if grid is not None else pred.grid(row)
            win, push, loss = mk.ah_probs(grid, line, "H")
            if win + loss > 0:
                pe = win / (win + loss)
                sa = get_odds(row, [c["sharp_home"], c["sharp_away"]])
                m = devig_power(sa if sa is not None else oa)[0]
                out["ah"] = {"line": line, "model": pe, "market": m, "final": w * m + (1 - w) * pe,
                             "scale": win + loss, "odds": haircut(oa, b.get("odds_haircut", 0.0))}
    return out


def candidates(pr: dict, cfg) -> list[dict]:
    b = cfg["betting"]
    cands = []

    def add(market, venue, sel, line, o, p, scale=1.0, min_edge=None):
        if not (b["min_odds"] <= o <= b["max_odds"]):
            return
        e = scale * (p * o - 1.0)
        if e >= (min_edge if min_edge is not None else b["min_edge"]):
            cands.append({"market": market, "venue": venue, "selection": sel, "line": line, "odds": float(o),
                          "prob": float(p), "edge": float(e), "scale": float(scale)})

    x2 = pr["x2"]
    for venue, odds in x2["venues"].items():
        pre = "PM:" if venue == "polymarket" else ""
        me = cfg["polymarket"]["min_edge"] if venue == "polymarket" else None
        for i, o in enumerate(odds):
            add(pre + "1X2", venue, OUT[i], None, o, x2["final"][i], min_edge=me)
    if "ou" in pr:
        o = pr["ou"]
        add("OU25", "bookmaker", "O", 2.5, o["odds"][0], o["final"])
        add("OU25", "bookmaker", "U", 2.5, o["odds"][1], 1 - o["final"])
    if "ah" in pr:
        a = pr["ah"]
        add("AH", "bookmaker", "H", a["line"], a["odds"][0], a["final"], a["scale"])
        add("AH", "bookmaker", "A", a["line"], a["odds"][1], 1 - a["final"], a["scale"])
    return cands


def best_bets(pr: dict, cfg, div=None) -> list[dict]:
    c = sorted(candidates(pr, cfg), key=lambda x: -x["edge"])
    out = []
    for bet in c:
        if bet["edge"] > _edge_cap(cfg, div):
            _log_suspicious(cfg, bet, div)   # too good to be true: flag for review, never publish
            continue
        out.append(bet)
        if len(out) >= cfg["betting"].get("max_bets_per_match", 1):
            break
    return out

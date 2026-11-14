"""Phase 3 self-evaluation loop (additive).

- Weekly report: Brier, log loss, calibration table, ROI/yield, max drawdown, CLV — overall and
  segmented by league, market type, odds range and edge bucket.
- Auto-adjustment within SAFE BOUNDS ONLY: per-segment stake multiplier REDUCTIONS (never
  increases) on negative CLV over 100+ settled picks; per-segment PAUSE when calibration error
  exceeds the limit. Recalibration of probabilities is applied post-blend, per segment, once
  >= 300 settled picks exist (isotonic on the segment's own settled picks).
- Every adjustment is logged with before/after values and the full registry state is snapshotted,
  so everything is reversible (segments-rollback).

Segment adjustments apply to the LIVE paper path only (never to the backtest: the registry is
built from live settled picks; feeding it back into a historical simulation would leak).
"""
from __future__ import annotations
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

SEG_DIMS = ("league", "market", "odds_range", "edge_bucket")
ODDS_EDGES = [1.0, 1.6, 2.5, 4.0, np.inf]
EDGE_EDGES = [0.0, 0.02, 0.05, 0.10, np.inf]
STATE_PATH = "models/segments.json"
SNAP_DIR = "models/segments_history"
ADJUST_LOG = "models/segments_log.jsonl"


def odds_range(odds: float) -> str:
    for lo, hi in zip(ODDS_EDGES[:-2], ODDS_EDGES[1:-1]):
        if lo <= odds < hi:
            return f"{lo:g}-{hi:g}"
    return "4+"


def edge_bucket(edge: float) -> str:
    for lo, hi in zip(EDGE_EDGES[:-2], EDGE_EDGES[1:-1]):
        if lo <= edge < hi:
            return f"{int(lo*100)}-{int(hi*100)}%"
    return "10%+"


def segment_key(row) -> tuple:
    return (row.get("div"), row.get("market"), odds_range(float(row["odds"])), edge_bucket(float(row["edge"])))


def _brier(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    return float(((df.prob - (df.pnl > 0).astype(float)) ** 2).mean())


def _logloss(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    p = np.clip(df.prob, 1e-9, 1)
    won = (df.pnl > 0).astype(float)
    return float(-(won * np.log(p) + (1 - won) * np.log(1 - p)).mean())


def _clv(df: pd.DataFrame) -> float:
    if df.empty or df.clv.isna().all():
        return float("nan")
    return float(df.clv.dropna().mean())


def calibration_bins(df: pd.DataFrame, bins: int = 8) -> list[dict]:
    if df.empty:
        return []
    p, hit = df.prob.to_numpy(), (df.pnl > 0).astype(float).to_numpy()
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi)
        if m.sum():
            out.append({"range": f"{lo:.2f}-{hi:.2f}", "n": int(m.sum()),
                        "predicted": float(p[m].mean()), "actual": float(hit[m].mean())})
    return out


def max_drawdown(df: pd.DataFrame, bankroll: float) -> float:
    if df.empty:
        return 0.0
    eq = bankroll + df.sort_values("ts").pnl.cumsum()
    return float(((eq - eq.cummax()) / eq.cummax()).min())


def segmented(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (league, market, odds range, edge bucket) segment with n, roi, clv, brier, ll, bias."""
    if df.empty:
        return pd.DataFrame(columns=["segment", "n", "roi", "clv", "brier", "ll", "bias"])
    df = df.copy()
    for col in ("stake", "pnl", "prob", "edge", "odds", "clv", "closing_odds", "p_market"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    rows = []
    for key, g in df.groupby([df["div"], df["market"], df["odds"].apply(odds_range), df["edge"].apply(edge_bucket)], dropna=False):
        won = (g.pnl > 0).astype(float)
        rows.append({"segment": "|".join(str(k) for k in key), "n": len(g),
                     "roi": float(g.pnl.sum() / g.stake.sum()) if g.stake.sum() else 0.0,
                     "clv": _clv(g), "brier": _brier(g), "ll": _logloss(g),
                     "bias": float(g.prob.mean() - won.mean())})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def overall(df: pd.DataFrame, bankroll: float) -> dict:
    if df.empty:
        return {"n": 0}
    return {"n": len(df), "staked": float(df.stake.sum()), "pnl": float(df.pnl.sum()),
            "roi": float(df.pnl.sum() / df.stake.sum()) if df.stake.sum() else 0.0,
            "brier": _brier(df), "ll": _logloss(df), "clv": _clv(df),
            "max_drawdown": max_drawdown(df, bankroll),
            "calibration": calibration_bins(df)}


def _settled(ledger) -> pd.DataFrame:
    """Settled bets with an actual result — withdrawn picks (pnl=0 by definition) are excluded.
    Numeric columns are coerced explicitly: pandas 3 reads SQLite columns containing NULLs as
    arrow-backed strings, which would break every aggregate."""
    b = ledger.settled_bets()
    if b.empty:
        return b
    b = b[b.status.isin(["won", "lost", "push"])].reset_index(drop=True)
    for col in ("stake", "pnl", "prob", "edge", "odds", "clv", "closing_odds", "p_market"):
        b[col] = pd.to_numeric(b[col], errors="coerce")
    for col in ("div", "market", "status", "selection"):   # plain objects: pandas 3 groupby chokes on arrow-string keys
        if col in b.columns:
            b[col] = b[col].astype(object)
    return b


def weekly_report(cfg: dict, ledger) -> str:
    """Markdown report; saved to reports/weekly_<date>.md. Returns the text."""
    bets = _settled(ledger)
    o = overall(bets, cfg["betting"]["bankroll"])
    L = [f"# Weekly report {dt.date.today().isoformat()}", ""]
    if not o["n"]:
        L.append("No settled bets yet.")
        return "\n".join(L)
    L += [f"Settled picks: {o['n']} | staked {o['staked']:.2f} | P&L {o['pnl']:+.2f} | ROI/yield {o['roi']:.2%}",
          f"Brier {o['brier']:.4f} | log loss {o['ll']:.4f} | **CLV {o['clv']:+.2%}** (primary skill signal) | max drawdown {o['max_drawdown']:.1%}",
          "", "## Calibration (probability buckets)", ""]
    L += [f"- {c['range']}: n={c['n']}, predicted {c['predicted']:.3f}, actual {c['actual']:.3f}" for c in o["calibration"]]
    L += ["", "## Segments (league | market | odds range | edge bucket)", ""]
    seg = segmented(bets)
    if seg.empty:
        L.append("- none")
    else:
        for _, r in seg.iterrows():
            clv = f"{r.clv:+.1%}" if np.isfinite(r.clv) else "n/a"
            L.append(f"- {r['segment']}: n={r.n}, ROI {r.roi:+.1%}, CLV {clv}, Brier {r.brier:.3f}, bias {r.bias:+.3f}")
    md = "\n".join(L)
    d = cfg["learning"].get("reports_dir", "reports")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"weekly_{dt.date.today().isoformat()}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return md


# ---------- safe auto-adjustment ----------
def _load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"segments": {}, "updated": None}


def _save_state(state: dict, cfg: dict):
    d = cfg["learning"]["models_dir"]
    state["updated"] = dt.datetime.utcnow().isoformat()
    os.makedirs(os.path.join(d, "segments_history"), exist_ok=True)
    with open(os.path.join(d, "segments_history", state["updated"].replace(":", "").replace(".", "") + ".json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    with open(os.path.join(d, "segments.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def _log(cfg: dict, entry: dict):
    d = cfg["learning"]["models_dir"]
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "segments_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def auto_adjust(cfg: dict, ledger) -> dict:
    """Reductions and pauses only, never increases. Returns the changes made."""
    L = cfg.get("learning", {}).get("segments", {})
    min_recal, min_clv_n = L.get("min_recal", 300), L.get("min_clv_n", 100)
    bias_limit = L.get("pause_bias", 0.05)
    bets = _settled(ledger)
    if bets.empty:
        return {"changed": 0}
    state = _load_state()
    changes = []
    for key, g in bets.groupby([bets["div"], bets["market"], bets["odds"].apply(odds_range), bets["edge"].apply(edge_bucket)], dropna=False):
        seg = "|".join(str(k) for k in key)
        cur = state["segments"].get(seg, {"multiplier": 1.0, "status": "active", "n": 0,
                                          "cal_kind": None, "cal": None, "updated": None})
        rec = {"ts": dt.datetime.utcnow().isoformat(), "segment": seg, "n": len(g),
               "clv": _clv(g), "bias": float(g.prob.mean() - (g.pnl > 0).mean()), "before": dict(cur)}
        won = (g.pnl > 0).astype(float)
        # 1) pause when calibration error exceeds the limit (checked first: paused segments skip the rest)
        if cur["status"] != "paused" and len(g) >= min_recal and abs(rec["bias"]) > bias_limit:
            cur.update(status="paused", updated=dt.datetime.utcnow().isoformat())
            rec.update(after=dict(cur), action="pause", reason=f"bias {rec['bias']:+.3f} beyond +/-{bias_limit}")
        else:
            # 2) recalibrate probabilities per segment (isotonic on the segment's own settled picks)
            if len(g) >= min_recal and len(set(won)) > 1:
                from sklearn.isotonic import IsotonicRegression
                iso = IsotonicRegression(out_of_bounds="clip").fit(g.prob.to_numpy(), won.to_numpy())
                cur.update(cal_kind="isotonic", cal=[float(v) for v in iso.predict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])],
                           updated=dt.datetime.utcnow().isoformat())
                rec.update(after=dict(cur), action="recalibrate", reason=">= 300 settled picks in segment")
            # 3) reduce the stake multiplier on negative CLV (never increase)
            if len(g) >= min_clv_n and np.isfinite(rec["clv"]) and rec["clv"] < 0:
                new_m = max(0.25, round(cur.get("multiplier", 1.0) * 0.5, 2))
                if new_m < cur.get("multiplier", 1.0):
                    cur.update(multiplier=new_m, updated=dt.datetime.utcnow().isoformat())
                    rec.update(after=dict(cur), action="reduce",
                               reason=f"CLV {rec['clv']:+.2%} negative over {len(g)} picks")
            cur["n"] = len(g)
            if "action" not in rec:
                rec["after"] = dict(cur)
        state["segments"][seg] = cur
        if rec.get("action"):
            _log(cfg, rec)
            changes.append(rec)
    if changes:
        _save_state(state, cfg)
    return {"changed": len(changes), "changes": changes}


def apply_segment(bet: dict, cfg: dict, div) -> dict:
    """Apply the segment registry to a candidate bet: probability recalibration (post-blend),
    stake multiplier, and pause. Mutates and returns the bet; paused segments set stake=0."""
    state = _load_state()
    seg = "|".join(str(k) for k in (div, bet["market"], odds_range(float(bet["odds"])), edge_bucket(float(bet["edge"]))))
    cur = state["segments"].get(seg)
    if not cur:
        return bet
    if cur.get("status") == "paused":
        bet["stake"] = 0.0
        bet["paused_segment"] = seg
        return bet
    cal, kind = cur.get("cal"), cur.get("cal_kind")
    if kind == "isotonic" and cal and len(cal) == 9:
        xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        p = float(np.interp(np.clip(bet["prob"], 0.0, 1.0), xs, cal))
        bet["prob"] = float(np.clip(p, 0.02, 0.98))
        bet["edge"] = bet["scale"] * (bet["prob"] * bet["odds"] - 1.0)
    m = cur.get("multiplier", 1.0)
    if m < 1.0:
        bet["stake"] = round(bet["stake"] * m, 2)
        bet["stake_multiplier"] = m
    return bet


def rollback(cfg: dict, ts: str):
    """Restore a snapshot from models/segments_history (reversible adjustments)."""
    d = cfg["learning"]["models_dir"]
    src = os.path.join(d, "segments_history", ts + ".json")
    if not os.path.exists(src):
        raise SystemExit(f"no snapshot {ts}")
    with open(src, encoding="utf-8") as f:
        state = json.load(f)
    with open(os.path.join(d, "segments.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    _log(cfg, {"ts": dt.datetime.utcnow().isoformat(), "action": "rollback", "to": ts})
    print(f"restored segments state from {ts}")

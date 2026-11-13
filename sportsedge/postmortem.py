"""Post-mortems: separate bad luck from real model error. Findings are only flagged when statistically significant."""
from __future__ import annotations
import datetime as dt, os
import numpy as np
import pandas as pd

IDX = {"H": 0, "D": 1, "A": 2}
NAMES = ["Home", "Draw", "Away"]


def _ll(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))


def rps(P, y):
    """Ranked probability score for ordered outcomes H<D<A (lower is better)."""
    O = np.eye(3)[y]
    return (((np.cumsum(P, 1) - np.cumsum(O, 1)) ** 2)[:, :2].sum(1) / 2).mean()


def _arrays(p: pd.DataFrame):
    y = p.outcome.map(IDX).to_numpy()
    return (p[["pm_h", "pm_d", "pm_a"]].to_numpy(), p[["pk_h", "pk_d", "pk_a"]].to_numpy(),
            p[["pf_h", "pf_d", "pf_a"]].to_numpy(), y)


def outcome_bias(PF, y, min_n=100):
    rows = []
    for k in range(3):
        pk = PF[:, k]
        exp, obs, var = pk.sum(), float((y == k).sum()), (pk * (1 - pk)).sum()
        z = (obs - exp) / np.sqrt(var) if var > 0 else 0.0
        rows.append({"outcome": NAMES[k], "expected": exp, "observed": obs, "z": z,
                     "flag": bool(len(y) >= min_n and abs(z) > 2.5)})
    return rows


def calibration_bins(PF, y, bins=8):
    p, hit = PF.ravel(), np.eye(3)[y].ravel()
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi)
        if m.sum():
            out.append({"range": f"{lo:.2f}-{hi:.2f}", "n": int(m.sum()), "predicted": float(p[m].mean()), "actual": float(hit[m].mean())})
    return out


def evaluate(preds: pd.DataFrame, min_n: int = 100) -> dict:
    if len(preds) == 0:
        return {"n": 0}
    PM, PK, PF, y = _arrays(preds)
    d = {"n": len(y), "ll_final": float(_ll(PF, y).mean()), "ll_market": float(_ll(PK, y).mean()),
         "ll_model_raw": float(_ll(PM, y).mean()),
         "rps_final": float(rps(PF, y)), "rps_market": float(rps(PK, y)), "bias": outcome_bias(PF, y, min_n), "calibration": calibration_bins(PF, y)}
    by = {}
    for div, g in preds.groupby("div"):
        _, pk, pf, yy = _arrays(g)
        by[div] = {"n": len(g), "ll_final": float(_ll(pf, yy).mean()), "ll_market": float(_ll(pk, yy).mean())}
    d["by_league"] = by
    return d


def bet_review(bets: pd.DataFrame) -> dict:
    """Was the P&L explained by variance, or is something wrong? z-score of actual vs expected profit."""
    if len(bets) == 0:
        return {"n": 0}
    stake, odds, p = bets.stake.to_numpy(), bets.odds.to_numpy(), bets.prob.to_numpy()
    exp_pnl = (stake * (p * odds - 1)).sum()
    var = (stake ** 2 * odds ** 2 * p * (1 - p)).sum()
    act = bets.pnl.sum()
    z = (act - exp_pnl) / np.sqrt(var) if var > 0 else 0.0
    return {"n": len(bets), "staked": float(stake.sum()), "pnl": float(act), "expected_pnl": float(exp_pnl),
            "z": float(z), "roi": float(act / stake.sum()),
            "verdict": ("consistent with variance" if abs(z) < 2 else
                        "WORSE than the model's own expectation: investigate calibration/data" if z < 0 else
                        "better than expected (do not read this as skill without CLV)")}


def render(ev: dict, br: dict) -> str:
    L = [f"# Post-mortem {dt.date.today().isoformat()}", ""]
    if ev.get("n", 0) == 0:
        return "\n".join(L + ["No settled predictions yet."])
    L += [f"Settled predictions: {ev['n']}",
          f"Log loss: final {ev['ll_final']:.4f} | market {ev['ll_market']:.4f} | raw model {ev['ll_model_raw']:.4f}",
          f"RPS: final {ev['rps_final']:.4f} | market {ev['rps_market']:.4f}",
          "(final < market means the blend adds value over the market; otherwise the model is not earning its keep)", "",
          "## Outcome bias (expected vs observed)"]
    for b in ev["bias"]:
        L.append(f"- {b['outcome']}: expected {b['expected']:.1f}, observed {b['observed']:.0f}, z={b['z']:+.2f}" + ("  <-- FLAG" if b["flag"] else ""))
    L += ["", "## Calibration (predicted vs actual frequency)"]
    L += [f"- {c['range']}: n={c['n']}, predicted {c['predicted']:.3f}, actual {c['actual']:.3f}" for c in ev["calibration"]]
    L += ["", "## By league"] + [f"- {k}: n={v['n']}, final {v['ll_final']:.4f}, market {v['ll_market']:.4f}" for k, v in ev["by_league"].items()]
    L += ["", "## Bets"]
    if br.get("n", 0):
        L += [f"- {br['n']} settled, ROI {br['roi']:.1%}, P&L {br['pnl']:.1f} vs expected {br['expected_pnl']:.1f} (z={br['z']:+.2f})",
              f"- Verdict: {br['verdict']}"]
    else:
        L.append("- none settled yet")
    L += ["", "Note: league/odds-bucket slices are exploratory. The agent only auto-acts via the gated improve step."]
    return "\n".join(L)


def run(cfg, ledger) -> tuple[dict, dict, str]:
    ev = evaluate(ledger.settled_predictions())
    br = bet_review(ledger.settled_bets())
    md = render(ev, br)
    d = cfg["learning"].get("reports_dir", "reports")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"postmortem_{dt.date.today().isoformat()}.md"), "w") as f:
        f.write(md)
    return ev, br, md


def health(cfg, ledger) -> list[str]:
    """Reasons to PAUSE publishing. Empty list = healthy."""
    H, reasons = cfg["learning"]["health"], []
    p = ledger.settled_predictions().tail(H["window"])
    if len(p) >= H["min_predictions"]:
        _, PK, PF, y = _arrays(p)
        diff = _ll(PF, y) - _ll(PK, y)           # >0: we are worse than the market
        rng = np.random.default_rng(0)
        lo = np.percentile([rng.choice(diff, len(diff)).mean() for _ in range(1000)], 5)
        if diff.mean() > H["tolerance"] and lo > 0:
            reasons.append(f"blend worse than market over last {len(p)} matches (+{diff.mean():.3f} log loss)")
    # per-league calibration drift: a league where we are persistently worse than the market is a
    # sign something changed there (odds schema, team-name join, style shift) — pause and investigate.
    if len(p):
        for div, g in p.groupby("div"):
            if len(g) < H.get("min_league", 100):
                continue
            _, pk, pf, yy = _arrays(g)
            d = _ll(pf, yy) - _ll(pk, yy)
            rng = np.random.default_rng(0)
            lo = np.percentile([rng.choice(d, len(d)).mean() for _ in range(1000)], 5)
            if d.mean() > H["tolerance"] and lo > 0:
                reasons.append(f"{div}: blend worse than market over last {len(g)} matches (+{d.mean():.3f} log loss) — drift?")
    b = ledger.settled_bets()
    if len(b):
        eq = b.pnl.cumsum() / cfg["betting"]["bankroll"]
        dd = (eq - eq.cummax()).min() * 100
        if dd < -H["max_drawdown_pct"]:
            reasons.append(f"drawdown {dd:.1f}% beyond limit")
    return reasons

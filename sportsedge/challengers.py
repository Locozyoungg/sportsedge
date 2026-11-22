"""Autonomous research in SHADOW mode (Phase 4d).

A challenger is a small module in the `challengers/` package exposing:
    NAME: str
    fit(hist: pd.DataFrame, cfg: dict) -> object
    knows(model, row) -> bool
    probs(model, row) -> np.ndarray   # [H, D, A], leak-free

The runner logs every pre-match challenger probability (write-once), settles it against real
results, and compares it with the champion's logged FINAL predictions on the same matches.
Challengers are NEVER published or staked. Promotion is NOT automatic here: a challenger that
beats the champion on >= shadow_min_matches with a bootstrap CI above 0 is PROPOSED to the admin;
it may only be adopted through the same gated `learn` gate and the admin's approval.
"""
from __future__ import annotations
import datetime as dt
import importlib
import json
import os
import pkgutil

import numpy as np
import pandas as pd

from .postmortem import _ll  # log loss helper (arrays)

IDX = {"H": 0, "D": 1, "A": 2}


def discover() -> list:
    """Import every module in the challengers/ package and return them."""
    import challengers  # noqa: F401  (the package ships alongside sportsedge)
    out = []
    for m in pkgutil.iter_modules(challengers.__path__):
        mod = importlib.import_module(f"challengers.{m.name}")
        for need in ("NAME", "fit", "probs", "knows"):
            if not hasattr(mod, need):
                raise ImportError(f"challengers.{m.name} is missing {need}")
        out.append(mod)
    return out


def _log_path(cfg) -> str:
    return os.path.join(cfg["learning"]["models_dir"], "shadow_log.jsonl")


def _report_path(cfg) -> str:
    return os.path.join(cfg["learning"]["models_dir"], "shadow_report.json")


def run_shadow(cfg: dict, hist: pd.DataFrame, fixtures: pd.DataFrame) -> dict:
    """Log challenger probabilities for every upcoming fixture (write-once per challenger+match)."""
    if fixtures.empty:
        return {"logged": 0}
    os.makedirs(cfg["learning"]["models_dir"], exist_ok=True)
    lp = _log_path(cfg)
    seen = set()
    if os.path.exists(lp):
        for line in open(lp, encoding="utf-8"):
            if line.strip():
                e = json.loads(line)
                seen.add((e["challenger"], e["match_date"], e["home"], e["away"]))
    n = 0
    with open(lp, "a", encoding="utf-8") as f:
        for mod in discover():
            try:
                model = mod.fit(hist, cfg)
            except Exception as e:  # a broken challenger must never break the pipeline
                print(f"challenger {mod.NAME} failed to fit, skipped: {e}")
                continue
            for _, row in fixtures.iterrows():
                if not mod.knows(model, row):
                    continue
                key = (mod.NAME, str(row["Date"])[:10], row["HomeTeam"], row["AwayTeam"])
                if key in seen:
                    continue
                p = np.asarray(mod.probs(model, row), float)
                if len(p) != 3 or not np.all(np.isfinite(p)) or p.sum() <= 0:
                    continue
                f.write(json.dumps({"challenger": mod.NAME, "match_date": str(row["Date"])[:10],
                                    "home": row["HomeTeam"], "away": row["AwayTeam"],
                                    "ph": p[0], "pd": p[1], "pa": p[2],
                                    "ts": dt.datetime.now(dt.UTC).replace(tzinfo=None).isoformat()}) + "\n")
                n += 1
    return {"logged": n}


def _boot_ci_lo_gt0(diff: np.ndarray, ci: float, seed: int = 0, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(diff, len(diff)).mean() for _ in range(n)])
    return float(np.percentile(means, 100 * (1 - ci))), float(np.percentile(means, 100 * ci))


def settle_shadow(cfg: dict, hist: pd.DataFrame, ledger=None) -> dict:
    """Settle logged shadow predictions against results and compare with the champion on the same matches.

    The champion's reference is the FINAL blend logged in the predictions table (model + market + calibration).
    A raw challenger beating that blend on many matches is a strong signal — that is what gets proposed.
    """
    lp, rp = _log_path(cfg), _report_path(cfg)
    if not os.path.exists(lp):
        return {"n": 0}
    rows = [json.loads(l) for l in open(lp, encoding="utf-8") if l.strip()]
    if not rows:
        return {"n": 0}
    log = pd.DataFrame(rows)
    log["match_date"] = pd.to_datetime(log.match_date)
    hist = hist[["Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
    m = log.merge(hist, left_on=["match_date", "home", "away"], right_on=["Date", "HomeTeam", "AwayTeam"])
    champ = None
    if ledger is not None:
        try:
            preds = ledger.settled_predictions()
            if len(preds):
                champ = preds[["match_date", "home", "away", "pf_h", "pf_d", "pf_a", "pk_h", "pk_d", "pk_a"]]
                champ["match_date"] = pd.to_datetime(champ.match_date)
                m = m.merge(champ, on=["match_date", "home", "away"], how="left")
        except Exception as e:
            print("shadow: champion join failed, comparing challenger to market only:", e)
    L = cfg["learning"]
    min_n = L.get("shadow_min_matches", 300)
    out = {"n": len(m), "min_matches": min_n, "challengers": [], "ts": dt.datetime.now(dt.UTC).replace(tzinfo=None).isoformat()}
    for name, g in m.groupby("challenger"):
        y = g.FTR.map(IDX).to_numpy()
        if len(y) < 50:
            continue
        P = g[["ph", "pd", "pa"]].to_numpy()
        rec = {"challenger": name, "n": len(y), "ll_challenger": float(_ll(P, y).mean())}
        if champ is not None and "pf_h" in g.columns:
            g2 = g.dropna(subset=["pf_h", "pk_h"])
            if len(g2):
                y2 = g2.FTR.map(IDX).to_numpy()
                rec["n_with_champion"] = len(y2)
                rec["ll_champion_final"] = float(_ll(g2[["pf_h", "pf_d", "pf_a"]].to_numpy(), y2).mean())
                rec["ll_market"] = float(_ll(g2[["pk_h", "pk_d", "pk_a"]].to_numpy(), y2).mean())
                diff = _ll(g2[["pf_h", "pf_d", "pf_a"]].to_numpy(), y2) - _ll(g2[["ph", "pd", "pa"]].to_numpy(), y2)
                lo, hi = _boot_ci_lo_gt0(diff, L.get("shadow_ci", 0.9))
                rec["gain_vs_champion"] = float(diff.mean())
                rec["gain_ci"] = [lo, hi]
                rec["propose_promotion"] = bool(len(y2) >= min_n and diff.mean() > 0 and lo > 0)
                rec["reason"] = ("shadow challenger beat the champion's FINAL blend with CI above 0"
                                 if rec["propose_promotion"] else
                                 f"needs {min_n} matched shadow predictions before promotion can be proposed"
                                 if len(y2) < min_n else "no statistically reliable improvement over the champion")
        out["challengers"].append(rec)
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def propose_text(rep: dict) -> str:
    lines = ["<b>Shadow research report</b>", f"settled shadow predictions: {rep.get('n', 0)}"]
    for c in rep.get("challengers", []):
        lines.append(f"\n<b>{c['challenger']}</b>: n={c['n']} shadow ll {c.get('ll_challenger', 0):.4f}")
        if "ll_champion_final" in c:
            lines.append(f"  champion ll {c['ll_champion_final']:.4f} | market ll {c['ll_market']:.4f} "
                         f"| gain {c['gain_vs_champion']:+.4f} CI {c['gain_ci']}")
            if c["propose_promotion"]:
                lines.append(f"  <b>PROPOSED FOR PROMOTION</b> — {c['reason']}. Approval + the gated learn still required.")
            else:
                lines.append(f"  {c['reason']}")
    return "\n".join(lines)

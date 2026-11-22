"""A6: rerun the walk-forward baseline per league on the fixed data (refit every 28 days),
plus the shadow challenger comparison. Prints the new baseline for Part B gating.
Usage: python scripts/a6_baseline.py   (read-only: touches no ledger, sends no Telegram)"""
from __future__ import annotations
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from sportsedge import data, learning
from sportsedge.backtest import run_backtest, summarize
from sportsedge.challengers import propose_text, run_shadow, settle_shadow
from sportsedge.cli import load_cfg
from sportsedge.ledger import Ledger


def fmt_bins(bins) -> str:
    return "  calib[conf->acc]: " + " ".join(
        f"{b['conf']:.2f}->{b['acc']:.2f}({b['n']})" for b in bins if b["n"])


def main() -> None:
    cfg = learning.apply_champion(load_cfg("config.yaml"))
    cfg["backtest"]["refit_days"] = 28   # the A6 protocol: refit every 28 days
    hist = data.load_history(cfg["paths"]["raw_dir"])
    report = {"champion": cfg.get("_version"), "model": cfg["model"], "leagues": {}}
    print(f"== A6 baseline, champion {cfg.get('_version')} ==", json.dumps(cfg["model"]))
    for div, g in hist.groupby("Div"):
        g = g.sort_values("Date")
        res = run_backtest(g, cfg)
        print(f"\n=== {div} (n={len(g)}) ===")
        print(summarize(res))
        print(fmt_bins(res.get("calibration_bins", [])))
        report["leagues"][div] = {k: v for k, v in res.items() if k != "bets"}
    print("\n== shadow challenger comparison ==")
    fx = data.load_fixtures(cfg["paths"]["raw_dir"], cfg)
    logged = run_shadow(cfg, hist, fx)
    rep = settle_shadow(cfg, hist, Ledger(cfg["paths"]["ledger"]))
    print(f"logged {logged['logged']} new shadow predictions; settled {rep.get('n', 0)}")
    if rep.get("challengers"):
        print(propose_text(rep))   # printed only, never sent
    with open("reports/a6_baseline.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nwrote reports/a6_baseline.json")


if __name__ == "__main__":
    main()

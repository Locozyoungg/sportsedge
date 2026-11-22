"""Part B evidence harness (read-only): per-league, walk-forward, no look-ahead.
B1 totals bias / B2 recalibration / B3 Poisson grid / B4 market weight / B7 poisson-vs-blend / B8 segments.
Usage: python scripts/b_eval.py   (writes nothing except prints; ~30-60 min on a slow machine)"""
from __future__ import annotations
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sportsedge import data, learning, markets as mk
from sportsedge.calibration import Calibrator
from sportsedge.cli import load_cfg
from sportsedge.model import PoissonModel

IDX = {"H": 0, "D": 1, "A": 2}


def ll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean())


def by_league(r, df):
    divs = df.loc[r["rid"], "Div"].to_numpy()
    return {d: {"idx": np.where(divs == d)[0]} for d in np.unique(divs)}


def main() -> None:
    cfg = load_cfg("config.yaml")
    df = data.load_history(cfg["paths"]["raw_dir"]).sort_values("Date")
    wf = learning.walk_forward(df, 200, 0.5, cfg)
    L = by_league(wf, df)

    print("\n===== B7 + B4: poisson walk-forward per league, market-weight sweep =====")
    print(f"{'league':<5}{'n':>6}{'ll_model':>10}{'ll_mkt':>10}{'best_w':>8}{'ll_blend@best':>14}")
    per_league = {}
    for d, v in L.items():
        i = v["idx"]
        pm, pk, y = wf["PM"][i], wf["PK"][i], wf["y"][i]
        llm, llk = ll(pm, y), ll(pk, y)
        best = min(((w, ll(w * pk + (1 - w) * pm, y)) for w in np.arange(0.1, 1.0, 0.1)),
                   key=lambda t: t[1])
        per_league[d] = {"n": len(y), "ll_model": llm, "ll_market": llk, "best_w": best[0], "ll_blend": best[1]}
        print(f"{d:<5}{len(y):>6}{llm:>10.4f}{llk:>10.4f}{best[0]:>8.1f}{best[1]:>14.4f}")

    print("\n===== B2: recalibration on a chronological holdout per league =====")
    print(f"{'league':<5}{'ll_identity':>12}{'ll_parametric':>14}{'ll_isotonic':>13}")
    from sklearn.isotonic import IsotonicRegression
    for d, v in L.items():
        i = v["idx"]
        pm, pk, y = wf["PM"][i], wf["PK"][i], wf["y"][i]
        cut = int(len(y) * 0.7)
        tr_p, tr_y = pm[:cut], y[:cut]
        ho_p, ho_y = pm[cut:], y[cut:]
        ids = ll(ho_p, ho_y)
        par = Calibrator.fit(tr_p, tr_y).apply(ho_p)
        ll_par = ll(par, ho_y)
        iso = np.zeros_like(ho_p)
        for k in range(3):
            ir = IsotonicRegression(out_of_bounds="clip").fit(tr_p[:, k], (tr_y == k).astype(float))
            iso[:, k] = ir.predict(ho_p[:, k])
        iso = np.clip(iso, 1e-4, None)
        iso = iso / iso.sum(axis=1, keepdims=True)
        ll_iso = ll(iso, ho_y)
        print(f"{d:<5}{ids:>12.4f}{ll_par:>14.4f}{ll_iso:>13.4f}")

    print("\n===== B1: totals bias per league (70/30 split, fit on train only) =====")
    print(f"{'league':<5}{'bias_uncorrected':>18}{'bias_corrected':>16}{'totals_ll':>11}")
    for d, g in df.groupby("Div"):
        g = g.sort_values("Date")
        cut = int(len(g) * 0.7)
        tr, ho = g.iloc[:cut], g.iloc[cut:]
        m = PoissonModel(200, 0.5, 10, True, False).fit(tr)
        b = mk.fit_ou_bias(m, tr)
        preds_u, preds_c, acts = [], [], []
        tll = []
        for _, r in ho.iterrows():
            if not m.knows(r.HomeTeam, r.AwayTeam):
                continue
            grid = m.score_grid(r.HomeTeam, r.AwayTeam)
            p = mk.ou_probs(grid, 2.5)[0]
            preds_u.append(p)
            preds_c.append(mk.apply_ou_bias(p, b))
            acts.append(1.0 if int(r.FTHG) + int(r.FTAG) > 2.5 else 0.0)
            tot = np.add.outer(np.arange(11), np.arange(11))
            pt = np.array([grid[tot == k].sum() for k in range(21)])
            tll.append(-np.log(np.clip(pt[int(r.FTHG) + int(r.FTAG)], 1e-9, 1)))
        bias_u = float(np.mean(preds_u) - np.mean(acts))
        bias_c = float(np.mean(preds_c) - np.mean(acts))
        print(f"{d:<5}{bias_u:>18.4f}{bias_c:>16.4f}{float(np.mean(tll)):>11.4f}")

    print("\n===== B3: coarse Poisson grid (walk-forward log loss per league, lower better) =====")
    print(f"{'league':<5}{'hl120r.2':>9}{'hl120r.5':>9}{'hl120r1':>9}{'hl200r.2':>9}{'hl200r.5':>9}{'hl200r1':>9}{'hl320r.5':>9}{'dc_off':>9}")
    configs = [(120, 0.2), (120, 0.5), (120, 1.0), (200, 0.2), (200, 0.5), (200, 1.0), (320, 0.5)]
    results = {}
    for hl, reg in configs:
        results[(hl, reg)] = learning.walk_forward(df, hl, reg, cfg)
    dc_off = None
    cfg_off = dict(cfg)
    cfg_off["model"] = dict(cfg["model"], dc=False)
    dc_off = learning.walk_forward(df, 200, 0.5, cfg_off)
    for d, v in L.items():
        row = []
        for hl, reg in configs:
            r = results[(hl, reg)]
            i = np.where(df.loc[r["rid"], "Div"].to_numpy() == d)[0]
            row.append(f"{ll(r['PM'][i], r['y'][i]):>9.4f}")
        i = np.where(df.loc[dc_off["rid"], "Div"].to_numpy() == d)[0]
        row.append(f"{ll(dc_off['PM'][i], dc_off['y'][i]):>9.4f}")
        print(f"{d:<5}" + "".join(row))

    print("\n===== B8: segments where the model is worse than the market (1X2, walk-forward) =====")
    for d, v in L.items():
        i = v["idx"]
        pm, pk, y = wf["PM"][i], wf["PK"][i], wf["y"][i]
        odds = 1 / np.max(pk, axis=1)   # implied favourite odds bucket
        for lo, hi in [(1.0, 1.7), (1.7, 2.2), (2.2, 3.5), (3.5, 99.0)]:
            j = np.where((odds >= lo) & (odds < hi))[0]
            if len(j) < 30:
                continue
            dll = ll(pm[j], y[j]) - ll(pk[j], y[j])
            flag = "  <-- model WORSE" if dll > 0.005 else ""
            print(f"{d:<5} odds[{lo}-{hi}) n={len(j):>4}  dll={dll:+.4f}{flag}")

    with open("reports/b_eval.json", "w") as f:
        json.dump({"per_league": per_league}, f, indent=2, default=str)
    print("\nwrote reports/b_eval.json")


if __name__ == "__main__":
    main()

"""Walk-forward backtest over all enabled markets. Never trains on data from the future."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import markets as mk
from .ensemble import PoissonPredictor, StackPredictor, Stacker, FEATURES, BASE_FEATURES, _pmodel, build_table
from .odds import devig_power, get_odds
from .risk import stake, DailyGuard, ExposureGuard
from .strategy import probabilities, best_bets

IDX = {"H": 0, "D": 1, "A": 2}


def _logloss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()


def run_backtest(df: pd.DataFrame, cfg: dict, seed: int = 0, table: pd.DataFrame | None = None) -> dict:
    df = df.sort_values("Date")
    mcfg, bcfg, kc = cfg["model"], cfg["betting"], cfg["backtest"]
    if len(df) <= kc["min_train_matches"]:
        return {"error": "not enough matches"}
    use_stack = mcfg.get("type") == "stack"
    if use_stack and table is None:
        table, _ = build_table(df, cfg)
    cur, last = df["Date"].iloc[kc["min_train_matches"]], df["Date"].max()
    bankroll = bcfg["bankroll"]
    bets, pm_l, pf_l, pk_l, y_l = [], [], [], [], []
    day, guard, expo = None, None, None
    step = kc.get("refit_days", kc.get("refit_every_days", 7))
    while cur <= last:
        end = cur + pd.Timedelta(days=step)
        win = df[(df.Date >= cur) & (df.Date < end)]
        if len(win):
            model = _pmodel(cfg).fit(df, as_of=cur)
            ou_b = mk.fit_ou_bias(model, df[df.Date < cur])
            pred = PoissonPredictor(model, ou_b)
            if use_stack:
                tr = table[(table.Date < cur)].dropna(subset=["pois_h"])
                tr = tr[tr.y >= 0]
                if len(tr) >= cfg.get("stack", {}).get("min_train", 300):
                    stacker = Stacker(cfg).fit(tr, tr.y)
                    pred = StackPredictor(model, stacker, lambda r: table.loc[r.name, BASE_FEATURES].to_dict(), ou_b)
            for rid, row in win.iterrows():
                row = row.copy(); row.name = rid
                if row.Date != day:
                    day = row.Date
                    guard = DailyGuard(bankroll, bcfg["daily_loss_limit_pct"])
                    expo = ExposureGuard(bankroll, bcfg["max_match_pct"], bcfg["max_day_pct"])
                pr = probabilities(pred, row, cfg)
                if pr is None:
                    continue
                y = IDX[row.FTR]
                x2 = pr["x2"]
                pm_l.append(x2["model"]); pf_l.append(x2["final"]); pk_l.append(x2["market"]); y_l.append(y)
                for bet in best_bets(pr, cfg):
                    if not guard.allowed():
                        break
                    s = stake(bankroll, bet["prob"], bet["odds"], bcfg, bet["scale"])
                    s = expo.clip((row.HomeTeam, row.AwayTeam), s)
                    if s <= 0:
                        continue
                    p = mk.pnl(bet["market"], bet["selection"], bet["line"], int(row.FTHG), int(row.FTAG), s, bet["odds"])
                    clv = np.nan
                    close = get_odds(row, bcfg["closing_cols"])
                    if close is not None and bet["market"] == "1X2":
                        clv = bet["odds"] * devig_power(close)[IDX[bet["selection"]]] - 1
                    bankroll += p
                    guard.record(p)
                    bets.append({**bet, "date": row.Date, "match": f"{row.HomeTeam} v {row.AwayTeam}",
                                 "stake": s, "pnl": p, "won": p > 0, "clv": clv, "bankroll": bankroll})
        cur = end
    b = pd.DataFrame(bets)
    out = {"bets": b}
    if y_l:
        y = np.array(y_l)
        out.update(logloss_model=_logloss(np.array(pm_l), y), logloss_market=_logloss(np.array(pk_l), y),
                   logloss_blend=_logloss(np.array(pf_l), y), n_predictions=len(y))
    if len(b):
        out.update(n_bets=len(b), staked=b.stake.sum(), profit=b.pnl.sum(), roi=b.pnl.sum() / b.stake.sum(),
                   hit_rate=b.won.mean(), avg_clv=b.clv.mean())
        peak = b.bankroll.cummax()
        out["max_drawdown"] = float(((b.bankroll - peak) / peak).min())
        rng = np.random.default_rng(seed)
        ratio = (b.pnl / b.stake).to_numpy()
        boots = [rng.choice(ratio, len(ratio)).mean() for _ in range(2000)]
        out["roi_ci95"] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        out["by_market"] = {m: {"n": len(g), "roi": float(g.pnl.sum() / g.stake.sum())} for m, g in b.groupby("market")}
    return out


def summarize(res: dict) -> str:
    if "error" in res:
        return res["error"]
    lines = [f"predictions scored : {res.get('n_predictions', 0)}"]
    if "logloss_model" in res:
        lines += [f"log loss  model/market/blend : {res['logloss_model']:.4f} / {res['logloss_market']:.4f} / {res['logloss_blend']:.4f}",
                  "  (lower is better; if model > market, the model alone is NOT beating the market)"]
    if "n_bets" in res:
        lo, hi = res["roi_ci95"]
        lines += [f"bets               : {res['n_bets']}  hit rate {res['hit_rate']:.1%}",
                  f"profit / ROI       : {res['profit']:.2f} / {res['roi']:.2%}   95% CI [{lo:.2%}, {hi:.2%}]",
                  f"avg CLV (1X2)      : {res['avg_clv']:.2%}  (persistently > 0 is the real sign of skill)",
                  f"max drawdown       : {res['max_drawdown']:.1%}",
                  "by market          : " + ", ".join(f"{m} n={v['n']} ROI {v['roi']:.1%}" for m, v in res["by_market"].items())]
        if lo <= 0 <= hi:
            lines.append("VERDICT: ROI is statistically indistinguishable from zero. Do not bet real money.")
    else:
        lines.append("No bets passed the filters.")
    return "\n".join(lines)

"""Walk-forward backtest over all enabled markets. Never trains on data from the future."""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
from . import markets as mk
from .ensemble import PoissonPredictor, StackPredictor, Stacker, FEATURES, BASE_FEATURES, _pmodel, build_table
from .odds import devig_power, get_odds
from .risk import stake, DailyGuard, allocate
from .strategy import probabilities, best_bets

IDX = {"H": 0, "D": 1, "A": 2}


def _logloss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()


def _simulate(df: pd.DataFrame, cfg: dict, seed: int = 0, table: pd.DataFrame | None = None) -> dict:
    """One walk-forward simulation: per-league model refit per window, picks, Kelly stakes, caps, settle.
    Returns predictions log loss + the bet log. Shared by the blend strategy and the market-only baseline."""
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
    day, guard, day_pending, halted = None, None, [], False
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

            def flush():
                """Settle one day's pending bets, scaling stakes down proportionally when caps bind."""
                nonlocal bankroll
                pending = [dict(match=(r.HomeTeam, r.AwayTeam), league=r.Div, market=b2["market"], stake=s2)
                           for r, b2, s2 in day_pending]
                allocate(pending, bankroll, bcfg)
                for (r, b2, _), p in zip(day_pending, pending):
                    s = p["stake"]
                    if s <= 0:
                        continue
                    pnl = mk.pnl(b2["market"], b2["selection"], b2["line"], int(r.FTHG), int(r.FTAG), s, b2["odds"])
                    clv = np.nan
                    close = get_odds(r, bcfg["closing_cols"])
                    if close is not None and b2["market"] == "1X2":
                        clv = b2["odds"] * devig_power(close)[IDX[b2["selection"]]] - 1
                    bankroll += pnl
                    guard.record(pnl)
                    bets.append({**b2, "date": r.Date, "match": f"{r.HomeTeam} v {r.AwayTeam}",
                                 "stake": s, "pnl": pnl, "won": pnl > 0, "clv": clv, "bankroll": bankroll})
                day_pending.clear()

            for rid, row in win.iterrows():
                row = row.copy(); row.name = rid
                if row.Date != day:
                    if day_pending:
                        flush()   # settle the previous day's bets before judging its losses
                    if guard and not guard.allowed():
                        halted = True   # previous day's realized loss exceeded the limit: halt, like the live path
                    day = row.Date
                    guard = DailyGuard(bankroll, bcfg["daily_loss_limit_pct"])
                if halted:
                    continue
                pr = probabilities(pred, row, cfg)
                if pr is None:
                    continue
                y = IDX[row.FTR]
                x2 = pr["x2"]
                pm_l.append(x2["model"]); pf_l.append(x2["final"]); pk_l.append(x2["market"]); y_l.append(y)
                for bet in best_bets(pr, cfg, row.get("Div")):
                    s = stake(bankroll, bet["prob"], bet["odds"], bcfg, bet["scale"])
                    if s <= 0:
                        continue
                    day_pending.append((row, bet, s))
            if day_pending:
                flush()
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


def run_backtest(df: pd.DataFrame, cfg: dict, seed: int = 0, table: pd.DataFrame | None = None) -> dict:
    """Phase 4: the configured strategy AND the market-only baseline (de-vigged consensus vs bookmaker
    prices, same filters and staking), simulated with the same walk-forward windows."""
    out = _simulate(df, cfg, seed, table)
    if "error" in out:
        return out
    cfg_m = copy.deepcopy(cfg)
    cfg_m["model"]["market_weight"] = 1.0   # probabilities = de-vigged market; the honest betting baseline
    m = _simulate(df, cfg_m, seed, table)
    bm = m.get("bets", pd.DataFrame())
    if len(bm):
        out.update(n_bets_market=len(bm), profit_market=float(bm.pnl.sum()),
                   roi_market=float(bm.pnl.sum() / bm.stake.sum()) if bm.stake.sum() else 0.0,
                   hit_rate_market=float(bm.won.mean()), max_drawdown_market=_drawdown(bm))
        rng = np.random.default_rng(seed)
        ratio = (bm.pnl / bm.stake).to_numpy()
        boots = [rng.choice(ratio, len(ratio)).mean() for _ in range(2000)]
        out["roi_ci95_market"] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return out


def _drawdown(b: pd.DataFrame) -> float:
    peak = b.bankroll.cummax()
    return float(((b.bankroll - peak) / peak).min())


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
    if "n_bets_market" in res:
        mlo, mhi = res["roi_ci95_market"]
        lines += [f"MARKET-ONLY baseline: {res['n_bets_market']} bets, ROI {res['roi_market']:.2%} "
                  f"95% CI [{mlo:.2%}, {mhi:.2%}], hit rate {res['hit_rate_market']:.1%}, "
                  f"max DD {res['max_drawdown_market']:.1%}",
                  "  (if the blend does not beat this baseline on log loss and ROI, the model adds no value)"]
    else:
        lines.append("MARKET-ONLY baseline: no bets passed the filters (the de-vigged consensus rarely "
                     "disagrees with the bookmaker by min_edge — worth knowing).")
    return "\n".join(lines)

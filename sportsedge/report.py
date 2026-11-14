"""Formats analysis posts for the premium channel. HTML parse mode; everything escaped."""
from __future__ import annotations
from html import escape as e

DISCLAIMER = ("<i>Analysis only, not guaranteed. Past results do not predict future ones. "
              "18+. Bet only what you can afford to lose. If betting stops being fun, take a break.</i>")


def pick_label(b: dict) -> str:
    m, sel = b.get("market", "1X2").replace("PM:", ""), b["selection"]
    if m == "OU25":
        return "Over 2.5 goals" if sel == "O" else "Under 2.5 goals"
    if m == "AH":
        ln = b["line"] if sel == "H" else -b["line"]
        team = b["home"] if sel == "H" else b["away"]
        return f"{e(team)} {ln:+g} (Asian handicap)"
    return {"H": f"{e(b['home'])} to win", "D": "Draw", "A": f"{e(b['away'])} to win"}[sel]


def confidence(edge: float) -> str:
    """Honest flag: HIGH only for edges well above the usual noise floor."""
    return "HIGH" if edge >= 0.08 else "STANDARD"


def format_bet(b: dict) -> str:
    imp = 1 / b["odds"]
    book = f" <i>({e(b['book'])})</i>" if b.get("book") else ""
    lines = [
        f"<b>{e(b['home'])} vs {e(b['away'])}</b>  <i>({e(str(b.get('div', '')))} · {str(b['date'])[:10]})</i>",
        f"Pick: <b>{pick_label(b)}</b> @ <b>{b['odds']:.2f}</b>{book}"
        + (" <i>(Polymarket YES)</i>" if b.get("venue") == "polymarket" else ""),
        f"Our probability {b['prob']:.0%} vs odds-implied {imp:.0%} → edge <b>{b['edge']:.1%}</b>"
        f"  <i>(confidence: {confidence(b['edge'])})</i>",
    ]
    if b.get("p_model") is not None and b.get("p_market") is not None:
        lines.append(f"Model {b['p_model']:.0%} · market (de-vigged) {b['p_market']:.0%}")
    if b.get("odds_ts"):
        lines.append(f"Odds snapshot: <i>{b['odds_ts'][:16].replace('T', ' ')}</i>")
    if b.get("injury_impact"):
        lines.append(f"⚠️ Lineup caveat: combined injury impact {b['injury_impact']:.1f}")
    if b.get("xg"):
        lines.append(f"Expected goals {b['xg'][0]:.2f} – {b['xg'][1]:.2f}; likeliest scores "
                     + ", ".join(f"{s} ({p:.0%})" for s, p in b.get("top_scores", [])))
    lines.append(f"Suggested stake: <b>{b['stake_pct']:.1f}%</b> of bankroll (fractional Kelly, capped)")
    return "\n".join(lines)


def format_digest(bets: list[dict], channel: str = "Premium Picks", day: str = "") -> str:
    if not bets:
        return (f"<b>{e(channel)}</b> {e(day)}\nNo bets today. Nothing cleared our value threshold, "
                f"and skipping is part of the edge.\n\n{DISCLAIMER}")
    head = f"<b>{e(channel)}</b> {e(day)}\n{len(bets)} value pick(s) today\n"
    body = "\n\n".join(format_bet(b) for b in sorted(bets, key=lambda x: -x["edge"]))
    return f"{head}\n{body}\n\n{DISCLAIMER}"


def format_track_record(cfg: dict, ledger, days: int = 90) -> str:
    """Phase 5: the weekly public track-record post — record, ROI, CLV, calibration, drawdown.
    Includes every settled result in the window (wins AND losses). No cherry-picking."""
    import pandas as pd
    from .eval import overall
    bets = ledger.settled_bets()
    if not len(bets):
        return f"<b>Track record</b>\nNo settled picks yet.\n\n{DISCLAIMER}"
    window = bets[pd.to_datetime(bets.match_date) >= pd.Timestamp.now().normalize() - pd.Timedelta(days=days)]
    use = window if len(window) else bets
    o = overall(use, cfg["betting"]["bankroll"])
    out = [f"<b>📊 Track record — last {days} days</b>", ""]
    if not o.get("n"):
        return f"<b>📊 Track record</b>\nNo settled picks in the window.\n\n{DISCLAIMER}"
    clv = f"{o['clv']:+.1%}" if pd.notna(o["clv"]) else "n/a"
    out += [f"Settled picks: {o['n']} | ROI/yield {o['roi']:+.1%} | P&L {o['pnl']:+.2f}",
            f"Closing-line value: <b>{clv}</b> (the honest skill measure) | max drawdown {o['max_drawdown']:.1%}",
            f"Brier {o['brier']:.3f} | log loss {o['ll']:.3f}", "",
            "<i>Calibration (predicted vs actual hit rate):</i>"]
    for c in o["calibration"][:6]:
        out.append(f"· {c['range']}: {c['predicted']:.0%} vs {c['actual']:.0%} (n={c['n']})")
    by_league = use.groupby("div").apply(lambda g: float(g.pnl.sum() / g.stake.sum()) if g.stake.sum() else 0.0)
    if len(by_league):
        out.append("")
        out.append("<i>ROI by league:</i> " + " · ".join(f"{k}: {v:+.1%}" for k, v in by_league.items()))
    out.append("")
    out.append(DISCLAIMER)
    out.append("<i>Full audited history: every pick, result and odds snapshot is in the public record.</i>")
    return "\n".join(out)


def format_results(rows, days: int, bankroll: float, totals, clv=None, drawdown=None) -> str:
    """rows: (date, home, away, sel, odds, stake, status, pnl). Publishes EVERY result, wins and losses."""
    out = [f"<b>Results: last {days} days</b>"]
    for d, h, a, sel, odds, stake, status, pnl in rows:
        icon = "✅" if status == "won" else "❌"
        out.append(f"{icon} {e(h)} v {e(a)} · {sel} @ {odds:.2f} · {pnl / bankroll * 100:+.2f}% of bankroll")
    if rows:
        tot = sum(r[7] for r in rows) / bankroll * 100
        out.append(f"\nPeriod: {len(rows)} bets, {tot:+.2f}% of bankroll")
    n, staked, pnl = totals
    if n:
        out.append(f"All-time (tracked): {n} bets, ROI {pnl / staked:.1%}")
    if clv is not None:
        out.append(f"Closing-line value over the period: {clv:+.2f}% (the honest measure of skill)")
    if drawdown is not None:
        out.append(f"Max drawdown over the period: {drawdown:.1%}")
    out.append(f"\n{DISCLAIMER}")
    return "\n".join(out)

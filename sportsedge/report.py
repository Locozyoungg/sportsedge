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


def format_bet(b: dict) -> str:
    imp = 1 / b["odds"]
    book = f" <i>({e(b['book'])})</i>" if b.get("book") else ""
    lines = [
        f"<b>{e(b['home'])} vs {e(b['away'])}</b>  <i>({e(str(b.get('div', '')))} · {str(b['date'])[:10]})</i>",
        f"Pick: <b>{pick_label(b)}</b> @ <b>{b['odds']:.2f}</b>{book}"
        + (" <i>(Polymarket YES)</i>" if b.get("venue") == "polymarket" else ""),
        f"Our probability {b['prob']:.0%} vs odds-implied {imp:.0%} → edge <b>{b['edge']:.1%}</b>",
    ]
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

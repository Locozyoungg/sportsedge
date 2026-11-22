from __future__ import annotations
import argparse, datetime as dt, json, os, yaml, pandas as pd
from . import data, learning, postmortem, markets as mk
from .backtest import run_backtest, summarize
from .broker import PaperBroker
from .ensemble import make_predictors
from .external import load_provider
from .ledger import Ledger
from .report import format_bet, format_digest, format_results
from .risk import stake, allocate
from .strategy import probabilities, best_bets
from .telegram import TelegramPublisher
from .validate import validate_history


def load_cfg(p="config.yaml"):
    with open(p) as f:
        return learning.apply_champion(yaml.safe_load(f))


def admin():
    """Diagnostics go to YOUR private chat (TELEGRAM_ADMIN_CHAT_ID), never to the paying channel."""
    return TelegramPublisher(chat_env="TELEGRAM_ADMIN_CHAT_ID")


def cmd_fetch(cfg):
    data.fetch(cfg["leagues"], cfg["seasons"], cfg["paths"]["raw_dir"])
    rep = validate_history(data.load_history(cfg["paths"]["raw_dir"]), cfg)
    print("validation:", "OK" if rep["ok"] else "FATAL", rep["fatal"], "| warnings:", rep["warn"])
    return rep


def _assert_valid(cfg, df):
    """Every engine command aborts on fatal data problems instead of training on bad rows."""
    rep = validate_history(df, cfg)
    if not rep["ok"]:
        raise SystemExit("data validation failed: " + "; ".join(rep["fatal"]))
    return rep


def cmd_backtest(cfg):
    df = data.load_history(cfg["paths"]["raw_dir"])
    _assert_valid(cfg, df)
    for div, g in df.groupby("Div"):
        print(f"\n=== {div} ({cfg['model'].get('type', 'poisson')}) ===")
        print(summarize(run_backtest(g, cfg)))


def cmd_odds(cfg):
    from .oddsfeed import fetch_live_fixtures
    hist = data.load_history(cfg["paths"]["raw_dir"])
    fx = fetch_live_fixtures(cfg, hist)
    if fx.empty:
        print("no live fixtures")
        return
    fx.to_csv(f"{cfg['paths']['raw_dir']}/fixtures_live.csv", index=False)
    print(f"wrote {len(fx)} live fixtures")


def cmd_polymarket_scan(cfg):
    from .polymarket import PolymarketClient, scan
    raw = cfg["paths"]["raw_dir"]
    fx = data.load_fixtures(raw, cfg)
    client = PolymarketClient(cfg)
    out = scan(fx, client.events(), client, cfg)
    out.to_csv(f"{raw}/fixtures_live.csv", index=False)
    n = int(out[["PM_H", "PM_D", "PM_A"]].notna().any(axis=1).sum())
    print(f"Polymarket prices attached to {n} of {len(out)} fixtures")


def cmd_polymarket_geocheck(cfg):
    from .polymarket import check_geoblock
    ok, info = check_geoblock(cfg)
    print("ALLOWED" if ok else "NOT ALLOWED / UNKNOWN (live trading stays off):", info)


def _brokers(cfg, ledger):
    paper = PaperBroker(ledger)
    pm = paper
    if cfg["polymarket"].get("live", {}).get("enabled"):
        try:
            from .polymarket import PolymarketLiveBroker
            pm = PolymarketLiveBroker(cfg, ledger)
        except Exception as e:  # any gate failing => stay on paper
            print("Polymarket LIVE disabled, using paper:", e)
    return paper, pm


def _match_impact(cfg: dict, ext, bet: dict) -> float:
    """Summed injury impact for the match (Phase 2 gate + Phase 5 caveat share this)."""
    return ext.impact(bet["home"], dt.date.today()) + ext.impact(bet["away"], dt.date.today())


def _lineup_gate(cfg: dict, ext, bet: dict) -> bool:
    """Phase 2 item 7: pre-publish lineup/injury gate. Returns True when the pick may be published.
    Impact scale: ~1 = a key starter out, 2+ = several key players out."""
    thresh = float(cfg.get("external", {}).get("publish_impact_threshold", 2.0))
    impact = _match_impact(cfg, ext, bet)
    bet["injury_impact"] = impact   # Phase 5: shown as a caveat on the post
    if impact >= thresh:
        d = cfg.get("learning", {}).get("models_dir", "models")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "publish_gate_log.jsonl"), "a") as f:
            f.write(json.dumps({"ts": dt.datetime.now(dt.UTC).replace(tzinfo=None).isoformat(), "home": bet["home"], "away": bet["away"],
                                "market": bet.get("market"), "selection": bet.get("selection"),
                                "impact": impact, "threshold": thresh, "action": "not_published"}, default=str) + "\n")
        print(f"lineup gate: NOT publishing {bet['home']} v {bet['away']} ({bet.get('selection')}) "
              f"— injury impact {impact:.1f} >= {thresh}")
        return False
    return True


def cmd_predict(cfg, record: bool, publish: bool = False):
    if (record or publish) and os.path.exists("STOP"):
        print("STOP file present: autopilot paused. Read it, fix the cause, then delete STOP.")
        return
    if publish and os.getenv("ODDS_API_KEY"):
        try:  # re-check prices just before posting; never fail the publish over a stale feed
            cmd_odds(cfg)
        except SystemExit:
            pass
    raw = cfg["paths"]["raw_dir"]
    hist, fx = data.load_history(raw), data.load_fixtures(raw, cfg)
    _assert_valid(cfg, hist)
    if fx.empty:
        print("no upcoming fixtures. Run: python -m sportsedge fetch  (or `odds` with ODDS_API_KEY)")
        return
    provider = load_provider(cfg)
    preds = make_predictors(hist, cfg, provider)
    ledger = Ledger(cfg["paths"]["ledger"]) if record else None
    paper, pm_broker = _brokers(cfg, ledger) if record else (None, None)
    b = cfg["betting"]
    bankroll = b["bankroll"]
    odds_ts = dt.datetime.fromtimestamp(os.path.getmtime(data.fixtures_path(raw))).isoformat()  # Phase 2 item 6: odds freshness
    slips, n, shown = [], 0, 0
    collected = []   # (date, bet) — allocated per day after all picks for that day are known
    for _, row in fx.sort_values("Date").iterrows():
        pred = preds.get(row["Div"])
        pr = probabilities(pred, row, cfg) if pred is not None else None
        if pr is None:
            continue
        x2 = pr["x2"]
        shown += 1
        if ledger:
            ledger.log_prediction(row["Div"], row, x2, cfg.get("_version", "v0"))
        print(f"{str(row.Date)[:10]} {row['Div']} {row.HomeTeam} v {row.AwayTeam} | H/D/A " + "/".join(f"{x:.2f}" for x in x2["final"]))
        for bet in best_bets(pr, cfg, row["Div"]):
            s = stake(bankroll, bet["prob"], bet["odds"], b, bet["scale"])
            if s <= 0:
                continue
            bet["stake"] = s
            if record or publish:   # segment registry is live paper-path state, not part of the backtest
                from .eval import apply_segment
                bet = apply_segment(bet, cfg, row["Div"])
                if bet["stake"] <= 0 or bet["edge"] < b["min_edge"]:
                    if bet.get("paused_segment"):
                        print(f"   ~~ segment paused [{bet['paused_segment']}]: skipping {row.HomeTeam} v {row.AwayTeam}")
                    continue
                s = bet["stake"]
            grid = pred.grid(row)
            top = sorted(((f"{i}-{j}", float(grid[i, j])) for i in range(4) for j in range(4)), key=lambda t: -t[1])[:3]
            sel = bet["selection"]
            pm_sel = None   # devigged market probability for the selection (Phase 3 ledger)
            p_model = None  # calibrated model probability for the selection (Phase 5 posts)
            if bet["market"] in ("1X2", "PM:1X2"):
                pm_sel = float(x2["market"]["HDA".index(sel)])
                p_model = float(x2["model"]["HDA".index(sel)])
            elif bet["market"] == "OU25" and "ou" in pr:
                pm_sel = float(pr["ou"]["market"] if sel == "O" else 1 - pr["ou"]["market"])
                p_model = float(pr["ou"]["model"] if sel == "O" else 1 - pr["ou"]["model"])
            elif bet["market"] == "AH" and "ah" in pr:
                pm_sel = float(pr["ah"]["market"] if sel == "H" else 1 - pr["ah"]["market"])
                p_model = float(pr["ah"]["model"] if sel == "H" else 1 - pr["ah"]["model"])
            bet.update(stake=s, home=row.HomeTeam, away=row.AwayTeam, date=row.Date, div=row["Div"],
                       xg=pred.xg(row), top_scores=top, odds_ts=odds_ts, p_market=pm_sel, p_model=p_model,
                       book=row.get(f"Book{sel}") if bet["market"] in ("1X2", "OU25") and sel in "HDAOU" else None,
                       ref=row.get(f"PM_TOKEN_{sel}") if bet["venue"] == "polymarket" else None)
            collected.append((row.Date, bet))
    # realized daily-loss limit (Phase 2 item 5): block all new bets if today's settled losses exceed it
    if record and ledger:
        loss_today = sum(r[7] for r in ledger.settled_since(1) if r[7] < 0)
        if loss_today < -bankroll * b["daily_loss_limit_pct"]:
            print(f"daily loss limit hit (today {loss_today:.2f}): placing nothing")
            collected = []
    from itertools import groupby
    for day, day_bets in groupby(sorted(collected, key=lambda t: t[0]), key=lambda t: t[0]):
        bets = [bet for _, bet in day_bets]
        pending = [dict(match=(bet["home"], bet["away"]), league=bet["div"], market=bet["market"], stake=bet["stake"])
                   for bet in bets]
        allocate(pending, bankroll, b)
        for bet, p in zip(bets, pending):
            bet["stake"] = p["stake"]
        for bet in bets:
            s = bet["stake"]
            if s <= 0:
                continue
            bet["stake_pct"] = 100 * s / bankroll
            sel = bet["selection"]
            print(f"   >> BET [{bet['market']}] {sel}{'' if bet['line'] is None else ' ' + format(bet['line'], '+g')} @ {bet['odds']:.2f} "
                  f"stake {s} edge {bet['edge']:.1%} p={bet['prob']:.2f}")
            if record:
                broker = pm_broker if bet["venue"] == "polymarket" else paper
                if broker.place(bet):
                    n += 1
                    slips.append(bet)   # only NEW bets are published, so reruns never double-post
            else:
                slips.append(bet)
    if shown == 0:
        print(f"no fixtures matched configured leagues {cfg['leagues']} "
              f"(fixtures cover: {sorted(fx.Div.dropna().unique().tolist())}). Try `odds` with ODDS_API_KEY.")
    if record:
        print(f"\nrecorded {n} new bets")
    if publish:
        tg = TelegramPublisher()
        posted_now = []   # bets actually posted in THIS run (new picks + sweep) — the digest covers exactly these
        for bet in slips:  # one post per pick so each can be edited later
            if not _lineup_gate(cfg, provider, bet):
                continue   # lineup/news gate: downgrade, never publish
            ids = tg.post(format_bet(bet))
            if ids and ids[0] > 0:
                posted_now.append(bet)
                if record:
                    bid = ledger.find_bet(bet)
                    if bid:
                        ledger.set_message_id(bid, ids[0])
        # sweep: bets in the ledger that were never posted (dry run, earlier failure) — post them now
        if record:
            for row in ledger.unposted_bets(str(dt.date.today())):
                (bid, d, h, a, m, s, line, odds, stk, prob, edge, venue, div, p_market, p_model) = row
                bet = dict(home=h, away=a, date=d, div=div, market=m, selection=s, line=line,
                           odds=odds, stake=stk, prob=prob, edge=edge, venue=venue,
                           stake_pct=100 * stk / bankroll, p_market=p_market, p_model=p_model)
                if not _lineup_gate(cfg, provider, bet):
                    continue   # lineup/news gate: downgrade, never publish
                ids = tg.post(format_bet(bet))
                if ids and ids[0] > 0:
                    posted_now.append(bet)
                    ledger.set_message_id(bid, ids[0])
        if not posted_now:
            print("nothing new to publish")
        else:
            tg.send(format_digest(posted_now, cfg.get("telegram", {}).get("channel_name", "Premium Picks"),
                                  dt.date.today().isoformat()))


def cmd_withdraw(cfg):
    """Re-check posted open picks against the latest odds; edit the post and withdraw the pick if the edge is gone.
    Withdrawals are recorded in the ledger (status='withdrawn', pnl=0) and never settled."""
    raw = cfg["paths"]["raw_dir"]
    hist, fx = data.load_history(raw), data.load_fixtures(raw, cfg)
    preds = make_predictors(hist, cfg, load_provider(cfg))
    led = Ledger(cfg["paths"]["ledger"])
    tg = TelegramPublisher()
    keep = set()
    fx_keys = set()
    for _, row in fx.iterrows():
        fx_keys.add((str(row.Date)[:10], row.HomeTeam, row.AwayTeam))
        pred = preds.get(row["Div"])
        pr = probabilities(pred, row, cfg) if pred is not None else None
        if pr is None:
            continue
        for bet in best_bets(pr, cfg, row["Div"]):
            keep.add((str(row.Date)[:10], row.HomeTeam, row.AwayTeam,
                      bet["market"], bet["selection"], format(bet["line"], "+g") if bet["line"] is not None else ""))
    n = 0
    for (bid, d, h, a, m, s, line, odds, mid) in led.posted_open_bets():
        if (d, h, a) not in fx_keys:
            continue   # not in the feed anymore (kicked off?) — settle handles it; never void on a guess
        key = (d, h, a, m, s, format(line, "+g") if line is not None else "")
        if key in keep:
            continue
        label = f"{h} v {a} · {s} @ {odds:.2f} [{m}]"
        print(f"withdrawing (edge no longer clears the threshold): {label}")
        if mid > 0:
            tg.edit(mid, f"<b>⚠️ ODDS MOVED — pick withdrawn.</b> {label}\n"
                         f"Fresh prices no longer clear our value threshold. The pick is void (no stake).")
        led.withdraw(bid)
        n += 1
    print(f"withdrew {n} pick(s)")


def cmd_monthly(cfg, days: int = 30):
    """Monthly subscriber report: results, ROI, CLV, drawdown over the window."""
    led = Ledger(cfg["paths"]["ledger"])
    rows = led.settled_since(days)
    clv = None
    b = led.settled_bets()
    if len(b) and "clv" in b.columns:
        win = b[pd.to_datetime(b.match_date) >= pd.Timestamp.now().normalize() - pd.Timedelta(days=days)]
        if len(win) and win.clv.notna().any():
            clv = float(win.clv.mean())
    TelegramPublisher().send(format_results(rows, days, cfg["betting"]["bankroll"], led.report(),
                                            clv=clv, drawdown=_window_drawdown(b, days, cfg["betting"]["bankroll"])))


def _window_drawdown(bets, days: int, bankroll: float) -> float:
    """Peak-to-trough drawdown of the paper bankroll over the window, as a negative fraction."""
    if bets.empty:
        return 0.0
    w = bets[pd.to_datetime(bets.match_date) >= pd.Timestamp.now().normalize() - pd.Timedelta(days=days)]
    if w.empty:
        return 0.0
    eq = bankroll + w.sort_values("ts").pnl.cumsum()
    return float(((eq - eq.cummax()) / eq.cummax()).min())


def cmd_settle(cfg):
    hist = data.load_history(cfg["paths"]["raw_dir"])
    aliases = data._aliases_for(cfg["paths"]["raw_dir"])
    led = Ledger(cfg["paths"]["ledger"])
    led.settle_predictions(hist)
    from .odds import devig_power, get_odds
    b = cfg["betting"]
    for bid, d, home, away, market, sel, line, odds, stk in led.open_bets():
        m = hist[(hist.Date == pd.Timestamp(d)) & (hist.HomeTeam == data.canonical_team(home, aliases))
                 & (hist.AwayTeam == data.canonical_team(away, aliases))]
        if len(m):
            r = m.iloc[0]
            clv, closing = None, None
            if market == "1X2" and sel in "HDA":
                close = get_odds(r, b["closing_cols"])
                if close is not None:
                    dev = devig_power(close)
                    closing = float(dev["HDA".index(sel)])
                    clv = float(odds * closing - 1)
            elif market in ("OU25", "OU"):
                c = b["markets"]["ou"]
                close = get_odds(r, [c["sharp_over"], c["sharp_under"]])
                if close is not None:
                    dev = devig_power(close)
                    closing = float(dev[0] if sel == "O" else dev[1])
                    clv = float(odds * closing - 1)
            elif market == "AH":
                c = b["markets"]["ah"]
                close = get_odds(r, [c["sharp_home"], c["sharp_away"]])
                if close is not None:
                    dev = devig_power(close)
                    closing = float(dev[0] if sel == "H" else dev[1])
                    clv = float(odds * closing - 1)
            led.settle_pnl(bid, mk.pnl(market, sel, line, int(r.FTHG), int(r.FTAG), stk, odds), clv, closing)
    n, staked, pnl = led.report()
    print(f"settled bets {n}  staked {staked:.2f}  pnl {pnl:.2f}  ROI {(pnl / staked if staked else 0):.2%}")


def cmd_results(cfg, days: int):
    led = Ledger(cfg["paths"]["ledger"])
    TelegramPublisher().send(format_results(led.settled_since(days), days, cfg["betting"]["bankroll"], led.report()))


def cmd_postmortem(cfg):
    ev, br, md = postmortem.run(cfg, Ledger(cfg["paths"]["ledger"]))
    print(md)
    return ev, br


def cmd_health(cfg) -> bool:
    reasons = postmortem.health(cfg, Ledger(cfg["paths"]["ledger"]))
    if reasons:
        with open("STOP", "w") as f:
            f.write("\n".join(reasons))
        admin().send("<b>SportsEdge PAUSED</b>\n" + "\n".join(reasons) + "\nDelete the STOP file to resume after review.")
        print("PAUSED:", reasons)
        return False
    print("healthy")
    return True


def cmd_learn(cfg, force=False):
    last = learning.last_log(cfg)
    if last and not force and (dt.datetime.now(dt.UTC).replace(tzinfo=None) - dt.datetime.fromisoformat(last["ts"])).days < cfg["learning"]["every_days"]:
        print("learning not due yet")
        return None
    df = data.load_history(cfg["paths"]["raw_dir"])
    _assert_valid(cfg, df)
    d = learning.improve(cfg, df)
    print({k: v for k, v in d.items() if k not in ("champion_cal", "challenger_cal")})
    admin().send(f"<b>Learning run</b>: {d['decision']}. {d.get('reason', '')}")
    return d


def cmd_weekly(cfg):
    """Phase 3: weekly report + safe auto-adjustment (reductions/pauses only)."""
    from .eval import weekly_report, auto_adjust
    led = Ledger(cfg["paths"]["ledger"])
    md = weekly_report(cfg, led)
    print(md)
    adj = auto_adjust(cfg, led)
    lines = [f"<b>Segment auto-adjust</b>: {adj['changed']} change(s)"]
    for c in adj.get("changes", []):
        lines.append(f"- {c['segment']}: {c['action']} ({c['reason']})")
    admin().send("\n".join(lines) + "\nEvery change is logged and reversible (segments-rollback).")
    return adj


def cmd_segments_status(cfg):
    from .eval import _load_state
    print(json.dumps(_load_state(), indent=2, default=str))


def cmd_track_record(cfg, days: int = 90):
    """Phase 5: weekly public track-record post (record, ROI, CLV, calibration, drawdown — losses included)."""
    from .report import format_track_record
    TelegramPublisher().send(format_track_record(cfg, Ledger(cfg["paths"]["ledger"]), days))


def cmd_shadow(cfg):
    """Shadow research: log challenger probabilities for upcoming fixtures, settle old ones, propose promotions."""
    from .challengers import run_shadow, settle_shadow, propose_text
    raw = cfg["paths"]["raw_dir"]
    hist = data.load_history(raw)
    _assert_valid(cfg, hist)
    fx = data.load_fixtures(raw, cfg)
    logged = run_shadow(cfg, hist, fx)
    rep = settle_shadow(cfg, hist, Ledger(cfg["paths"]["ledger"]))
    print(f"logged {logged['logged']} new shadow predictions; settled {rep.get('n', 0)}")
    if rep.get("challengers"):
        admin().send(propose_text(rep))
    return rep


def cmd_autopilot(cfg):
    """Daily cron entrypoint: fetch+validate -> (live odds) -> (polymarket) -> settle -> post-mortem -> health -> learn -> shadow -> publish."""
    rep = cmd_fetch(cfg)
    if not rep["ok"]:
        with open("STOP", "w") as f:
            f.write("data validation failed: " + "; ".join(rep["fatal"]))
        admin().send("<b>SportsEdge PAUSED</b>: data validation failed: " + "; ".join(rep["fatal"]))
        return
    if os.getenv("ODDS_API_KEY"):
        cmd_odds(cfg)
    if cfg["polymarket"].get("scan"):
        try:
            cmd_polymarket_scan(cfg)
        except Exception as e:
            print("polymarket scan failed (continuing without it):", e)
    cmd_settle(cfg)
    cmd_postmortem(cfg)
    if not cmd_health(cfg):
        return
    if cmd_learn(cfg):
        cfg = load_cfg()
    try:
        cmd_shadow(cfg)
    except Exception as e:
        print("shadow research failed (continuing without it):", e)
    if dt.date.today().weekday() == 0:
        try:
            cmd_weekly(cfg)
        except Exception as e:
            print("weekly evaluation failed (continuing without it):", e)
    cmd_predict(cfg, record=True, publish=True)
    if dt.date.today().weekday() == 0:
        cmd_results(cfg, 7)
        cmd_track_record(cfg)


def main():
    from dotenv import load_dotenv
    load_dotenv()   # credentials live in .env (never committed); existing env vars take precedence
    ap = argparse.ArgumentParser(prog="sportsedge")
    ap.add_argument("cmd", choices=["fetch", "validate", "backtest", "predict", "paper", "settle", "publish", "results",
                                    "telegram-test", "postmortem", "health", "learn", "rollback", "autopilot",
                                    "odds", "polymarket-scan", "polymarket-geocheck", "withdraw", "monthly",
                                    "shadow", "weekly", "segments-status", "segments-rollback", "track-record"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--version", type=int)
    ap.add_argument("--ts", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    {"fetch": lambda: cmd_fetch(cfg), "validate": lambda: print(validate_history(data.load_history(cfg["paths"]["raw_dir"]), cfg)),
     "backtest": lambda: cmd_backtest(cfg), "predict": lambda: cmd_predict(cfg, False),
     "paper": lambda: cmd_predict(cfg, True), "publish": lambda: cmd_predict(cfg, True, True),
     "settle": lambda: cmd_settle(cfg), "results": lambda: cmd_results(cfg, a.days),
     "telegram-test": lambda: TelegramPublisher().send("<b>SportsEdge</b>: test message. Channel wiring works."),
     "postmortem": lambda: cmd_postmortem(cfg), "health": lambda: cmd_health(cfg),
     "learn": lambda: cmd_learn(cfg, a.force), "rollback": lambda: learning.rollback(cfg, a.version),
     "autopilot": lambda: cmd_autopilot(cfg), "odds": lambda: cmd_odds(cfg),
     "polymarket-scan": lambda: cmd_polymarket_scan(cfg), "polymarket-geocheck": lambda: cmd_polymarket_geocheck(cfg),
     "withdraw": lambda: cmd_withdraw(cfg), "monthly": lambda: cmd_monthly(cfg, a.days),
     "shadow": lambda: cmd_shadow(cfg), "weekly": lambda: cmd_weekly(cfg),
     "segments-status": lambda: cmd_segments_status(cfg),
     "segments-rollback": lambda: __import__("sportsedge.eval", fromlist=["rollback"]).rollback(cfg, a.ts),
     "track-record": lambda: cmd_track_record(cfg, a.days)}[a.cmd]()

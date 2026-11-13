# SportsEdge — build report (2026-11-13)

## What this repo contains (baseline v0.2)

A football (soccer) betting research agent: time-decayed Dixon-Coles Poisson model, Elo/form/xG-proxy/
rest/congestion features, stacked ensemble (logistic regression + histogram gradient boosting), 4-parameter
recalibrator, de-vigged Pinnacle market reference, markets (1X2, Over/Under 2.5, Asian handicap incl.
quarter-line settlement), paper ledger (SQLite, write-once prediction log), fractional-Kelly staking with
exposure/daily-loss caps, STOP kill-switch, Telegram publisher, post-mortems, and a gated autonomous
learning loop with champion registry, rollback, and promotion cooldown. ~2,500 lines, 32 tests.

## What was built in this session (Phase 4 + 4b + 4d completion)

| Item | Files |
|---|---|
| Streamlit dashboard: bankroll curve, drawdown, CLV with bootstrap CI, calibration bins, per-league ROI, open bets, prediction log, learning history, on-demand backtest, post-mortem archive | `sportsedge/dashboard.py` |
| Per-bet CLV captured at settlement from closing odds; `div` and `message_id` columns in the ledger (with migration for old ledgers) | `sportsedge/ledger.py`, `sportsedge/cli.py` |
| Publish flow: re-fetch live odds before posting (if `ODDS_API_KEY`), one Telegram post per pick, message ids stored, unposted-bet sweep (nothing double-posted, nothing silently lost) | `sportsedge/cli.py`, `sportsedge/telegram.py` |
| `withdraw` command: re-check posted open picks vs fresh odds, edit the exact post ("⚠️ ODDS MOVED — pick withdrawn"), void the pick (never settle a withdrawn bet; matches that left the feed are left for settlement) | `sportsedge/cli.py`, `sportsedge/ledger.py` |
| `monthly` subscriber report: results (wins AND losses), ROI, CLV, drawdown | `sportsedge/report.py`, `sportsedge/cli.py` |
| Shadow challenger research: `challengers/` package (contract: NAME, fit, knows, probs), write-once prediction log, settlement against real results, comparison vs the champion's FINAL blend on the same matches, promotion PROPOSAL only (never auto-promoted, never published or staked) | `sportsedge/challengers.py`, `challengers/quick.py` |
| Per-league drift detection in `health` (pauses publishing when one league is persistently worse than the market) | `sportsedge/postmortem.py` |
| Lock-guarded scheduler entrypoint, Dockerfile, GitHub Actions CI, `.gitattributes`, config keys (`learning.shadow_*`, `learning.health.min_league`) | `scripts/`, `Dockerfile`, `.github/` |
| Tests for all of the above | `tests/test_phase4.py` |

## Real-data verification

- `fetch`: 7,592 matches, 6 leagues (2023-08 → 2026-11), validation clean (no duplicates, no bad scores, odds sane).
- Live end-to-end test (2026-11-13): odds feed fetched 59 real fixtures across all 6 configured leagues,
  `predict` produced probabilities for every match and 6 value picks, `paper` recorded 6 bets + 59 write-once
  predictions, `shadow` logged 59 challenger predictions, `health` stayed healthy, `settle`/`postmortem`
  correctly found nothing to settle yet (matches are upcoming). Telegram posting fails gracefully until the
  chat ids are corrected (bot token verified OK).
- Two real bugs found by testing and fixed: (1) football-data `fixtures.csv` ships a latin-1-mojibake BOM
  (`ï»¿Div`) and lowercase `div` — both now normalized in `data.load_fixtures`/`_read` (regression-tested);
  (2) `predict` exited silently when no fixture matched the configured leagues — now reports it.
- `backtest` (walk-forward, per league): the honest verdict is **no evidence of edge yet**.
  - Model log loss is worse than the market in every league; the blend beats the raw model but only
    beats the market in 2 of 6 leagues and by a thin margin.
  - ROI is negative or statistically indistinguishable from zero in every league; CLV is negative
    everywhere with meaningful sample size.
  - **Verdict: paper trade only. Live stays locked (config + env + geoblock + manual approval).**

## Known follow-ups (not built — gated or need your decision)

- M-Pesa paid subscriptions (Phase 4c): needs Kenyan legal/BCLB check + your approval before any billing code.
- Polymarket live: adapter exists but is deliberately untested against the exchange; WebSocket order book,
  depth-aware sizing, and fee-aware edge are Phase 4e items.
- Real xG (Understat/FBref) instead of the shot proxy; player-availability and lineup features.
- `datetime.utcnow()` deprecation warnings (Python 3.13) — cosmetic, scheduled for removal upstream.

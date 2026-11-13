# SportsEdge v0.2: data-driven betting research agent

Football (soccer) first. Pipeline: free data -> time-decayed Poisson model -> blend with de-vigged
sharp market (Pinnacle) -> edge vs the odds you can actually get -> fractional-Kelly stake -> paper ledger.

## Quick start
    pip install -r requirements.txt
    python -m pytest -q                 # sanity tests (synthetic data)
    python -m sportsedge fetch          # download results + odds + upcoming fixtures
    python -m sportsedge backtest       # walk-forward, per league, with ROI CI and CLV
    python -m sportsedge predict        # today's slip (no recording)
    python -m sportsedge paper          # record paper bets
    python -m sportsedge settle         # settle paper bets from new results (+ CLV per 1X2 bet)

## Dashboard (Phase 4)
    streamlit run sportsedge/dashboard.py
Bankroll curve, drawdown, CLV (with bootstrap CI), calibration bins, per-league ROI, open bets,
prediction log, learning history, on-demand backtest, post-mortem archive. Reads only what the
pipeline writes (ledger, reports/, models/), so it works as soon as the first paper bets settle.

## Scheduler
    python scripts/run_autopilot.py                # one autopilot cycle, lock-guarded
    python scripts/run_autopilot.py --command paper
Run it daily from cron or Windows Task Scheduler (action: `python scripts/run_autopilot.py`,
start in the project folder; a non-zero exit code = failed run — alert on it). A lock file
prevents overlapping runs and a crashed run's lock is taken over after 6 hours.
Docker: `docker build -t sportsedge .` and run the same entrypoint on a schedule.

## Publishing (Phase 4b)
    python -m sportsedge publish        # place + post: re-fetches live odds first (if ODDS_API_KEY), one post per pick
    python -m sportsedge withdraw       # re-check posted open picks vs fresh odds: edit post + void pick if the edge is gone
    python -m sportsedge monthly --days 30   # subscriber report: results, ROI, CLV, drawdown
Each pick is its own Telegram post and its message_id is stored in the ledger, so `withdraw` can
edit the exact post ("⚠️ ODDS MOVED — pick withdrawn") and the pick settles as void (no stake).
Bets whose posts failed (or were dry-run) are swept and posted on the next `publish` — nothing is
double-posted, nothing is silently lost.

## Shadow research (Phase 4d)
    python -m sportsedge shadow         # log challenger predictions for upcoming fixtures + settle + propose
Drop a challenger module (NAME, fit, knows, probs -> [H,D,A]) in `challengers/` — see
`challengers/quick.py`. Every pre-match probability is logged write-once, settled against real
results, and compared to the champion's FINAL blend on the SAME matches. A challenger that beats
the champion on >= `learning.shadow_min_matches` settled matches with a bootstrap CI above 0 is
PROPOSED to your admin chat — it is never published, never staked, and promotion still requires
the gated `learn` gate plus your explicit approval. `autopilot` runs shadow daily.
Drift detection: `health` now also pauses publishing when a single league is persistently worse
than the market over its last 100+ settled predictions (odds-schema or team-name-join drift).

## Phase 2 safety & calibration (config.yaml)
- `betting.min_stake_pct` (0.5%): picks whose stake rounds below this share of bankroll are never placed or published.
- `betting.max_edge` / `betting.edge_caps` (0.15; per-league overrides): edges above the cap are
  "model-vs-market disagreement" — logged to `models/suspicious_log.jsonl`, never published.
- `betting.max_league_pct`, `max_market_pct`, `max_day_pct` (3%/3%/6%): per-day exposure caps; when a
  cap binds, ALL stakes in the group scale down proportionally. The daily loss limit is enforced in the live path.
- Totals calibration: a leak-free per-league correction on P(over 2.5) is fit at every model refit
  (AUDIT: Unders were over-predicted). Only corrects upward; never applied when training shows no bias.
- `external.publish_impact_threshold` (2.0): pre-publish lineup/injury gate — picks on matches at or
  above this impact are downgraded and logged (`models/publish_gate_log.jsonl`), never posted.
- Every pick records its odds snapshot time (`odds_ts`) and, from the odds feed, which book supplied
  the best price (`book`, shown in posts).
- Data: `fetch`/`load_history` normalize BOM-mojibake headers (all seasons now actually train), and
  `validate` fails on NaN `Div` rows.

## v0.2 model upgrades- **Dixon-Coles** low-score correction + time decay + optional new-team prior (`model.dc`, `model.newteam_prior`).
- **Stacked ensemble** (`model.type: stack`): Poisson probs + Elo + form + xG-proxy (from shots) + rest/congestion + market + injuries,
  fed to logistic regression + gradient boosting. All features are leak-free and walk-forward. It is only adopted by the held-out gate in `learn`.
- **Markets**: 1X2, Over/Under 2.5, Asian handicap (incl. quarter lines, pushes, correct settlement). BTTS probabilities available for analysis.
- **Execution realism**: multi-bookmaker live odds (`odds`, needs ODDS_API_KEY), `odds_haircut`, your own prices via
  `data/external/local_odds.csv`, per-match and per-day exposure caps, one bet per match by default.
- **Data safety**: `validate` (duplicates, bad scores, odds sanity) runs in `fetch`/`autopilot` and pauses on fatal problems.
- **Injuries/lineups**: drop a sourced CSV in `data/external/injuries.csv` (see the .example); every row needs a source URL.
- **Metrics**: log loss, RPS, calibration, CLV, per-market ROI with bootstrap CIs.

## Polymarket
    python -m sportsedge polymarket-geocheck   # is your location allowed? (fails closed; never use a VPN to bypass)
    python -m sportsedge polymarket-scan       # attach live YES prices (as decimal odds) to fixtures
    python -m sportsedge paper                 # strategy compares bookmaker AND Polymarket prices, picks the best edge
A YES share bought at ask c pays 1, so odds = 1/c and the same edge/Kelly maths applies. Polymarket needs a higher edge (config
`polymarket.min_edge`) to cover spread, fees and resolution risk, and skips thin or wide-spread books.
Live orders (`polymarket.live.enabled: true`, `pip install -r requirements-live.txt`, LIVE_CONFIRM, wallet key, geoblock pass,
manual approval, per-order USD cap) are implemented but UNTESTED against the real exchange. Paper trade first, then $1 orders.
Polymarket's availability rules change often and public sources disagree about Kenya: confirm on their official geoblock page,
and check Kenyan law/BCLB position on prediction markets before using real money.

## Telegram premium channel
1. In Telegram, message @BotFather -> /newbot -> copy the token.
2. Create a PRIVATE channel, add the bot as an administrator with "Post messages".
3. Get the channel id (forward a channel post to @RawDataBot, or use the -100... id) and put both in `.env`:
   TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID. Without them, posts print as a dry run.
4. Commands:
       python -m sportsedge telegram-test      # wiring check
       python -m sportsedge publish            # record new paper bets + post analysis (never double-posts)
       python -m sportsedge settle             # then
       python -m sportsedge results --days 7   # post ALL results, wins and losses
Posts include: pick, odds, our probability vs implied, edge, expected goals, likeliest scores, stake as % of bankroll.

## Autonomous learning loop
    python -m sportsedge autopilot     # run daily from cron: fetch > settle > post-mortem > health > learn > publish
    python -m sportsedge postmortem    # luck-vs-error report (calibration, bias, expected vs actual P&L)
    python -m sportsedge learn --force # challenger search + gated promotion
    python -m sportsedge rollback --version 1
How it works:
- Every pre-match prediction (not only bets) is logged write-once, then settled against real results.
- Post-mortems separate bad luck from model error using z-scores; nothing is "fixed" because of one loss.
- `learn` searches a small bounded space (half-life, shrinkage, market weight, 4-parameter calibration) on a TUNING period, then
  must beat the current champion on a LATER held-out period with a bootstrap CI above 0, with a 30-day promotion cooldown.
  Every version is saved and rollback-able; every decision is appended to models/learning_log.jsonl.
- `health` writes a STOP file and alerts your admin chat if the blend is significantly worse than the market or drawdown is too deep.
  While STOP exists nothing is published. Review, fix, delete STOP.
- It can never change risk limits, staking, or live flags, and it will often (correctly) decide "no reliable improvement".
- Diagnostics go to TELEGRAM_ADMIN_CHAT_ID (your private chat), never the paying channel.

## Honest expectations
- No model can guarantee accuracy. Bookmaker odds already contain most public information.
- The real success metric is **CLV** (beating the closing line) over hundreds of bets, not a lucky streak.
- If the backtest verdict says ROI is indistinguishable from zero, do not stake real money.
- Live execution is disabled and locked (config + LIVE_CONFIRM env + manual approval).
  Kenyan retail bookmakers have no official betting API; automating their sites breaks ToS. Place slips manually.
- Only bet money you can afford to lose. Betting can be addictive: set deposit limits.

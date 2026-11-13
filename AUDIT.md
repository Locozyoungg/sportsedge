# AUDIT — SportsEdge "Premium Picks" agent (Phase 1, read-only)

Date: 2026-11-14. No code was changed for this audit. All findings reference files in `sportsedge/`
and were verified against the code or against the real data in `data/raw/`.

---

## ⚠️ Finding 0 (new, discovered during the audit): silent multi-season data loss

`data.load_history()` (`data.py:38-46`) concatenates every `*.csv` in `data/raw` and trusts that
all files agree on column names. They do not:

| file | first header |
|---|---|
| `2324_D1.csv` | `Div` (clean) |
| `2425_D1.csv`, `2526_D1.csv`, `2627_D1.csv` | `﻿Div` (latin-1-mojibake BOM — fetched before the `_clean_cols` fix) |

pandas aligns the concat **by column name**, so the rows of the mojibake files land under a
different column and their `Div` becomes `NaN` — silently. Consequence, verified empirically:

- The concatenated frame has 7,592 rows but only **2,304 usable rows** (one season per league);
  the rest sit in `Div = NaN` rows that every `groupby("Div")` and `df[df.Div == ...]` drops.
- All models fit in `predict`/`paper`/`learn`/`backtest` — including the 6 paper picks currently
  in the ledger — were trained on **one season per league**, not 3.3 seasons.
- `validate_history()` (`validate.py:9-43`) checks column *presence* but not NaN in `Div`, so
  the earlier "validation: OK" was a false negative.
- D1/F1 walk-forward backtests scored only 9 predictions because of this (their 2324 file is the
  only clean one; everything later vanished).

**This is the highest-priority fix for Phase 2**: apply the existing `_clean_cols` in
`load_history` (one line), and make `validate_history` flag NaN `Div` rows (one line).

---

## a) How each probability is produced

| Market | Pipeline | Code |
|---|---|---|
| 1X2 | Dixon-Coles time-decayed Poisson → score grid → `predict_1x2` → optional 4-param calibrator → blend with devigged market at weight `market_weight` | `model.py:98-111`, `strategy.py:36-39` |
| O/U 2.5 | Same score grid → `ou_probs(grid, 2.5)` → blend with devigged O/U market | `strategy.py:47-51`, `markets.py:6-8` |
| Asian handicap | Same score grid → `ah_probs(grid, line)` (quarter lines split stake) → blend | `strategy.py:52-67`, `markets.py:21-29` |
| BTTS | `btts(grid)` — computed, analysis only, never bet | `markets.py:11-12` |

Alternative: `model.type: stack` (logistic regression + HGB on Elo/form/xG-proxy/rest/injuries/
market features) replaces the 1X2 probabilities; O/U and AH still come from the Poisson grid.
Default config is `poisson` (`config.yaml:7`).

## b) Are O/U and 1X2 derived from the same goal-rate model? Is there a low-goal bias?

**Yes, one grid drives everything** (`model.score_grid`, `model.py:98-107`; all markets read it).

Empirical walk-forward check on real data (per league, refit every 28 days, no look-ahead):

| league | n | pred P(over 2.5) | actual | bias | pred goals | actual goals |
|---|---|---|---|---|---|---|
| E0 | 82 | 0.628 | 0.659 | **−3.0pp** | 3.29 | 3.28 |
| E1 | 252 | 0.504 | 0.508 | −0.5pp | 2.72 | 2.68 |
| I1 | 80 | 0.475 | 0.513 | **−3.8pp** | 2.61 | 2.61 |
| SP1 | 80 | 0.463 | 0.513 | **−4.9pp** | 2.57 | 2.64 |
| D1, F1 | 9 | unreliable (n=9 — see Finding 0) | | | | |
| **all** | 503 | **0.515** | **0.549** | **−3.4pp** | 2.782 | 2.770 |

**Verdict: a systematic low-goal bias on the totals market — Unders are over-predicted by
~3.4pp overall (up to 5pp in SP1).** The mean-goal rate is essentially unbiased (2.782 vs 2.770):
the bias is **distributional** (Poisson under-dispersion around the 2.5 line), not a problem with
league goal rates, attack/defence shrinkage, or time decay. With `min_edge = 0.04`
(`config.yaml:32`), a −3.4pp model bias is enough to manufacture phantom value on Under 2.5 picks —
indeed the live run picked two Unders (Lazio U2.5 @ 2.48 "edge 31.6%", Union Berlin U2.5 @ 2.79
"edge 11.8%"). The right fix is a per-league totals calibration correction (Phase 2, item 4),
tuned on the (fixed) historical data, not by hand.

## c) How is edge computed?

`strategy.py:78`: `e = scale * (p * o - 1.0)` — exactly `probability × odds − 1`, where `p` is
the blended probability and `o` the venue's decimal odds; `scale` accounts for Asian-handicap
push fractions (`strategy.py:95-96`). Confirmed.

## d) Is the bookmaker margin removed before comparing? Which method?

Partially. The **market reference** probability is de-vigged with the power method
(`odds.py:11-27`, `devig_power`) — Pinnacle sharp columns preferred, then Polymarket mids, then
the best venue's own odds (`strategy.py:34-36`). The venue's **own price is not de-vigged**
before `edge = p*odds − 1`, which is correct — you bet at the raw price, and the margin is
"beaten" by being better than the de-vigged consensus. O/U and AH markets use the same
devig-first, compare-second pattern (`strategy.py:50, 65`).

## e) Which picks can show a 0.0% stake yet still get published?

- Truly zero stake is filtered: `if s <= 0: continue` (`cli.py:113`, same in `backtest.py:60`).
- **But there is no minimum stake threshold**, and `stake()` rounds to 2 decimals
  (`risk.py:9`), so a stake as low as 0.01 units is possible (0.001% of bankroll).
- The published post formats stake as `{stake_pct:.1f}%` (`report.py:31`), so **any stake below
  0.05% of bankroll displays as "0.0%"**. Live evidence: the 2026-11-13 publish included
  `Ein Frankfurt H @ 2.43 stake 0.13` — 0.013% of bankroll, displayed as "0.0%".
- The sweep path (`cli.py:146-156`) re-posts stored bets with no stake check at all (they were
  stored with `s > 0`, so this is display-level, but the minimum-stake rule should apply at
  placement).

## f) Are odds timestamped? How old can they be at publish time?

**No.** There is no odds timestamp anywhere:
- The ledger stores placement time `ts` but not the age/source-time of the odds
  (`ledger.py:13, 40-47`).
- `fixtures_live.csv` is a snapshot file; its mtime is the only freshness signal
  (`data.py:49-54`).
- `publish` re-fetches the feed when `ODDS_API_KEY` is set (`cli.py:84-88`), so published picks
  are as fresh as the feed run — minutes old. Without the key, `fixtures.csv` (site snapshot)
  is used and can be **days old with no check whatsoever**.
- Re-pricing at T−2h before kickoff (Phase 2, item 6) does not exist; `withdraw` (added this
  session) re-checks posted picks whenever it is run, but nothing schedules it.

## g) Is there a record of past picks, results, and closing odds?

Yes, mostly:
- **Bets**: `bets` table — match, market, selection, odds, stake, prob, edge, status, pnl,
  mode, line, venue, div, clv, message_id (`ledger.py:10-15`).
- **Predictions**: write-once `predictions` table — model/market/final probs + odds per match
  (`ledger.py:17-21, 68-75`).
- **Results**: settled against history in `cmd_settle` (`cli.py:166-183`).
- **Closing odds**: captured only as a CLV scalar for **1X2 bets** at settlement
  (`cli.py:174-181`); O/U and AH bets get **no CLV** and closing odds themselves are not stored.
- **Post-mortems** (reports/) and shadow logs (models/) add calibration/bias analytics.

---

## Phase 2 known issues — where each stands in the code

1. **Minimum stake / 0.0% display** — REPRODUCES. No threshold; `risk.py:6-9`, display at
   `report.py:31`; live example above.
2. **Suspicious-edge guard (edge > 15%)** — MISSING. `strategy.py:78` computes edge; nothing
   caps or flags it. Live run produced edges up to 31.6% on a top-five league (Lazio U2.5).
3. **Market blend 50/50** — ALREADY EXISTS. `strategy.py:18,38` (1X2), `:51` (OU), `:66` (AH);
   weight is `model.market_weight: 0.5` (`config.yaml:11`) and the gated `learn` loop tunes it
   on held-out data (`learning.py:168-184`). Per Phase 1 rule 4: this finding does not need a
   new implementation; Phase 3 can re-tune it with the fixed dataset.
4. **Totals bias check** — REPRODUCES as a real bias (see §b): −3.4pp overall on Over-2.5.
   Needs a per-league correction; evidence above.
5. **Correlation control** — PARTIAL. Per-match 3% and per-day 8% caps exist
   (`config.yaml:38-39`, `risk.py:25-37`), but: (i) no per-league or per-market-type caps;
   (ii) `ExposureGuard.clip` **truncates** the last stake instead of scaling all stakes down
   proportionally (`risk.py:31-37`); (iii) `DailyGuard` (realized-loss limit,
   `risk.py:12-22`) is used in the backtest (`backtest.py:47`) but **not in the live paper
   path** (`cli.py` never constructs it) — `daily_loss_limit_pct: 0.06` is currently a no-op in
   production.
6. **Odds freshness** — MISSING (see §f).
7. **Lineup/news gate** — MISSING for the default model. `CsvInjuryProvider` exists
   (`external.py:17-29`) but injury features only enter the **stack** model's feature table
   (`features.py:49`, `ensemble.py:10-11`); with `model.type: poisson` (the default),
   injuries affect nothing, and there is no pre-publish lineup check or downgrade logic.
8. **Line shopping** — PARTIAL. The odds feed already takes the **best price across all
   books** (`oddsfeed.py:32`), but the book name is not stored or shown in posts, and
   `local_odds.csv` overrides (`oddsfeed.py:92-103`) replace rather than compete with the feed.

---

## Readiness notes for later phases

- **Phase 3** (self-evaluation loop): the foundations exist — ledger with predictions,
  post-mortems with Brier/log-loss/calibration bins (`postmortem.py:49-62`), weekly/monthly
  posts, the gated `learn` loop with cooldown and rollback (`learning.py:124-209`). Missing
  per Phase 3 spec: segment-level (league × market × odds range × edge bucket) calibration,
  auto-reduction/pausing of bad segments, and CLV for non-1X2 markets.
- **Phase 4** (backtest validation): `backtest.py:18-88` is a proper walk-forward with
  log-loss vs market, ROI bootstrap CI, CLV and max drawdown. Its current numbers are
  **unreliable until Finding 0 is fixed** (trained on 1 season per league).
- **Phase 5** (publish format): posts include market, odds, model prob, edge, stake, xG,
  likeliest scores (`report.py:20-33`) but are missing: book name, market probability,
  confidence flag, odds timestamp, lineup caveat. The weekly results post exists
  (`cli.py cmd_results`); it includes losses and the responsible-gambling footer
  (`report.py:5-6`).

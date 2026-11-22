# SportsEdge Prediction Engine — Robustness Phase Report

Scope: prediction engine only (loader, validation, model, strategy, backtests). Publishing,
Telegram, dashboard, billing untouched. Date: 2026-11-22. All tests green (see §2).

## 0. Audit discrepancy (rule 7)

The input audit claimed "7,592 rows loaded, only 2,304 usable; `Div` NaN from mojibake BOM".
**Stale.** On the current data: 7,592/7,592 rows usable, 0 NaN `Div` — the BOM fix landed in
Phase 2 (`load_history` applies header cleaning, and today's `fetch` rewrote every raw file
through the cleaned path). Rows per league-season are all ≥ 143 (expectation "~3.3 seasons per
league" confirmed: partial 2023 + full 2024 + 2025 + partial 2026). The structural issues the
audit pointed at (two loader paths, silent date/row drops, weak validation) were real and are
fixed in A1/A2.

## 1. Part A — correctness (all merged, all tested)

| Task | Status | Files | Tests added |
|---|---|---|---|
| A1 single loader | done | `sportsedge/data.py` | 5 (mixed headers, dedupe, both date formats, loud failures, BOM bytes, lowercase fixtures) |
| A2 strong validation | done | `sportsedge/validate.py`, `sportsedge/cli.py`, `config.yaml` | 6 (new fatals, `_assert_valid` abort) |
| A3 team aliases | done (empty map: no true duplicates found) | `data/team_aliases.csv`, `sportsedge/data.py`, `sportsedge/cli.py` | 2 (canonicalisation + settle share) |
| A4 leakage & determinism | done (+1 real leak fixed) | `sportsedge/features.py`, `sportsedge/backtest.py`, `sportsedge/learning.py`, `sportsedge/ensemble.py`, `config.yaml` | 4 (feature no-leak, OOS stability, as-of fit, determinism) |
| A5 probability guards | done | `sportsedge/model.py`, `sportsedge/strategy.py`, `config.yaml` | 4 (DC non-negativity, tail-mass raise, clip+renorm, end-to-end) |
| A6 baseline rerun | done (this report's §3) | `scripts/a6_baseline.py`, `sportsedge/backtest.py` (Brier+calibration) | 1 |

Details that matter:

- **A1** — one `_normalize_frame` for every CSV: BOM/mojibake/whitespace header cleaning,
  canonical column names, robust dates (ISO, `dd/mm/yyyy`, `dd/mm/yy`, timestamps; raises on
  garbage, drops only blanks), dedupe on (Div, Date, HomeTeam, AwayTeam) per-file AND across
  files. `load_history` refuses to load files missing score columns, and no longer scans
  `fixtures_live.csv` as history (was a bug).
- **A2** — new fatals: missing Date, blank teams, implausible scores (`validate.max_goals: 15`),
  per-league-season rows below `validate.min_rows: 50`, overround outside [0.98, 1.25] for >2%
  of rows. `predict`, `paper`, `publish`, `backtest`, `learn`, `shadow` all abort on fatal
  validation via `_assert_valid`.
- **A3** — full audit of all leagues/seasons found **no true cross-season spelling variants**
  (fuzzy hits were different clubs; all single-season teams are genuine promotions/relegations).
  The aliases file ships empty and is wired into the loader + settle join so a future
  two-spelling club is fixed in one place.
- **A4** — **found and fixed a real leak**: `FeatureState` form/rest/xG took the last N records
  regardless of date and Elo mutated instantly, so out-of-order updates leaked into past
  queries. Form/xG/rest now filter by `date <= query`; Elo keeps an as-of history. Seeds are
  config (`backtest.seed`, `learning.seed`); two identical backtests are byte-identical.
  Unknown teams already fell back to league-average strengths (20th-percentile prior); now
  covered by a test.
- **A5** — `score_grid` raises when mass beyond `max_goals` exceeds `model.max_tail_mass`
  (default 0.20; real data max 4.3%), clips Dixon-Coles cells at 0 (the (0,0) cell went
  negative at λ·λ·ρ > 1 — silently possible before). `strategy._sane_probs` warns + clips +
  renormalises any non-finite/non-positive/non-summing probability set; applied to model,
  calibrated, market, blend 1X2 and O/U/AH finals.

## 2. Tests

Final suite: **84 passed** (was 61 at session start; +23 new). `tests/test_engine_robustness.py`
holds the 24 engine-robustness tests.

## 3. A6 baseline — fixed data, walk-forward, refit every 28 days, champion = stack v1 (market_weight 0.9)

| league | n pred | ll model | ll market | ll blend | Brier blend | bets | ROI (95% CI) | CLV (1X2) |
|---|---|---|---|---|---|---|---|---|
| D1 | 693 | 1.0063 | 0.9652 | 0.9653 | 0.1911 | 14 | −43.9% [−100%, +25%] | −1.78% |
| E0 | 930 | 1.0156 | 0.9818 | 0.9819 | 0.1954 | 8 | −10.4% [−100%, +57%] | +0.86% |
| E1 | 1505 | 1.0713 | 1.0345 | 1.0355 | 0.2076 | 21 | +17.9% [−55%, +66%] | −3.59% |
| F1 | 694 | 0.9945 | 0.9640 | 0.9633 | 0.1907 | 5 | +25.4% [−100%, +158%] | −1.65% |
| I1 | 941 | 0.9978 | 0.9657 | 0.9658 | 0.1917 | 11 | −19.2% [−100%, +73%] | −2.62% |
| SP1 | 941 | 0.9794 | 0.9509 | 0.9513 | 0.1877 | 9 | −44.1% [−100%, +74%] | −0.13% |

(All figures are the champion (stack v1, w=0.9) walk-forward run from
`reports/a6_baseline.json`.)

- The **model alone is worse than the market in every league** (log-loss and Brier).
- The blend at w=0.9 is statistically indistinguishable from the market (expected: 90% market).
- Bet counts are tiny; every ROI CI straddles zero → **no edge detectable**.
- Calibration bins are healthy (confidence tracks accuracy, slight mid-bin noise).
- Market-only baseline barely ever bets (de-vigged consensus rarely disagrees with the
  bookmaker by `min_edge` — worth knowing).

## 4. Part B — accuracy upgrades (each gated on out-of-sample evidence)

| Item | Verdict | Evidence |
|---|---|---|
| B1 totals dispersion | **already merged** (leak-free per-league logit shift, Phase 2); re-validated | Bias uncorrected→corrected: D1 −7.5→−7.0pp, E1 −10.4→−8.4pp, I1 −8.2→−5.5pp, SP1 −5.7→−5.7 (no train bias), E0 +0.8→+4.6pp (overshoot on test slice), F1 +3.5→+3.5. Negative-binomial **not merged**: no evidence of remaining OOS dispersion error beyond the shift, high cost/risk. |
| B2 1X2 recalibration | **rejected** | Parametric ≈ identity in all 6 leagues (Δll ≤ 0.008, mixed sign). Isotonic improves D1 (−0.005) but hurts E0/E1/I1 (+0.03/+0.04/+0.02) — overfits. Champion keeps identity calibration. |
| B3 Poisson grid | **no change** | Coarse grid re-searched: hl200/reg1.0 wins 4/6 leagues by ≤0.005 ll (poisson-alone). The learning grid already contains these points and the gated learn loop (which owns promotion, on the blend holdout) chose hl200/reg0.5. No manual override. |
| B4 market weight | **no change** | w-sweep per league: every league prefers w=0.9 (the max swept) — matches the learn loop's promotion. No per-league or per-market weight change has evidence. |
| B5 injuries | **left off** | Wiring merged behind `model.use_injuries: false` (attack-strength multiplier `exp(−k·impact)`, k=`injury_strength: 0.15`, impact capped at 5; predictor-level, applies to Poisson path). **Coverage: 0** — `data/external/injuries.csv` does not exist (only the .example), so the flag is a no-op today. Add sourced rows to the CSV to use it. |
| B6 real xG | **interface + blocker** | `XgProvider` interface added (`external.py`). Blocked: Understat has no official API (per-match HTML scraping, no schema/rate contract); FBref blocks scraping by robots/ToS and needs per-season team-id mapping. Shots-based `xg_proxy` remains. |
| B7 stack vs poisson | **stack kept** | Stack champion (w=0.9) ≈ market in all leagues; poisson-alone worse everywhere (§3). The learn loop's promotion stands on the fixed data. |
| B8 segments | **reported** | Model worse than market in **all 18 segments** (6 leagues × 3 odds buckets, n≥211 each; dll +0.012…+0.046). No segment where the model reliably beats the market. No auto-pause (per instructions). |

**No Part B item was enabled** — none improved out-of-sample log-loss with a consistent sign.
The engine's honest state: market ≈ blend, model adds nothing yet.

## 5. Plain-language verdict

On the fixed data (7,592 matches, 6 leagues, walk-forward, refit 28d): **the engine does not
beat the closing market out of sample, in any league or market.** The model alone is worse than
the de-vigged market everywhere (log-loss gap +0.01…+0.04 per league); the production blend is
90% market and therefore indistinguishable from it. Bets are few, ROI CIs all straddle zero,
and CLV is mixed/negative in 5 of 6 leagues. Sample size for the betting ledger (5–21 bets per
league) is far too small to conclude anything about ROI — but the log-loss evidence over
~700–1,500 predictions per league is decisive: **keep it paper-only**. Live trading stays
locked; this is consistent with BACKTEST_REPORT.md.

## 6. Out-of-scope findings (reported, not fixed)

1. `cmd_backtest` prints per-league results but has no `--refit` CLI switch; A6 protocol
   (28-day refit) is available via the config key. Cosmetic.
2. The learn loop's Poisson challenger path trains with `learning.refit_days: 28` windows but
   the `walk_forward` helper ignores `model.max_tail_mass`/`use_injuries` (PoissonModel built
   inline with defaults). Harmless today (injuries off), worth aligning if B5 is enabled.
3. `fixtures_live.csv` feeds `load_fixtures` with full timestamps; the date parser accepts
   them but strips time. The settle join uses dates only — no impact.
4. `datetime.utcnow()` warnings are gone; 2,668 deprecation warnings in tests were eliminated
   in the prior session's commit (068dbd1).

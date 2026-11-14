# Phase 4 — Backtest & validation report (2026-11-14)

Walk-forward simulation on the full corrected dataset (7,592 matches, 6 leagues, 2023-08 → 2026-11).
No look-ahead: per league, the model is refit every 7 days using only matches before the test window;
stakes follow the Phase 2 rules (fractional Kelly ≤ 2%, min stake 0.5%, per-match/league/market/day
caps with proportional scaling, daily-loss halt, suspicious-edge guard at 15%). Historical closing
odds (Pinnacle close) are used for CLV. The **market-only baseline** is the same pipeline with the
model removed: probabilities = de-vigged market (Pinnacle, power method), same filters and staking.

## 1X2 log loss (lower is better)

| league | model | market | blend (50/50) |
|---|---|---|---|
| D1 | 1.0079 | **0.9650** | 0.9809 |
| E0 | 0.9971 | **0.9827** | 0.9845 |
| E1 | 1.0609 | **1.0353** | 1.0435 |
| F1 | 0.9949 | **0.9639** | 0.9741 |
| I1 | 0.9889 | **0.9656** | 0.9719 |
| SP1 | 0.9772 | **0.9507** | 0.9603 |

**The model alone does not beat the de-vigged market in any league.** The blend sits between them,
as a 50/50 average must. Said plainly: the model adds no information over Pinnacle prices.

## Betting strategies (out-of-sample)

| league | blend bets | blend ROI | blend CLV (1X2) | blend max DD | market-only bets | market-only ROI |
|---|---|---|---|---|---|---|
| D1 | 338 | −15.8% | −7.3% | −43.9% | 2 | −23.0% (n=2, meaningless) |
| E0 | 425 | −5.7% | −5.3% | −29.9% | 0 | — |
| E1 | 709 | −5.2% | −5.8% | −30.9% | 9 | +16.7% (n=9, CI ±100%) |
| F1 | 366 | −1.6% | −6.2% | −23.1% | 1 | −100% (n=1) |
| I1 | 472 | −8.3% | −7.4% | −46.7% | 2 | +55.3% (n=2, meaningless) |
| SP1 | 448 | −9.4% | −6.4% | −47.0% | 0 | — |

## The verdict, plainly

1. **The claimed edges do not survive out of sample.** The blend placed 2,758 bets over 3 seasons;
   ROI is negative in every league and CLV (the honest skill signal) is −5% to −7% everywhere.
   The Phase 2 safety work halved drawdowns but cannot manufacture an edge that isn't there.
2. **The market itself almost never finds a qualifying bet** against B365 prices with a 4% edge
   threshold: 14 bets in 3 seasons across 6 leagues. That is itself the finding — a correct,
   disciplined system at min_edge 4% would publish almost nothing. The model's hundreds of
   "edges" are noise, not signal.
3. **Recommendation:** per the safety rules — publish fewer or no picks. Concretely: raise
   `min_edge` (e.g. 0.06–0.08) and/or switch the default blend weight toward the market
   (`market_weight` 0.7–0.8), then re-run this comparison before any real-money decision.
   Live trading remains locked; paper trading continues to accumulate the Phase 3 track record.
4. The Dixon-Coles + ensemble direction is not disproven as research infrastructure — the
   Phase 3 self-evaluation loop and shadow challengers now exist to test improvements with
   discipline. But as of today the data says: the model does not beat the market.

Reproduce: `python -m sportsedge backtest` (fetched data, current config).

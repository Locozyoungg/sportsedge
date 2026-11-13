# Changelog

All notable changes to SportsEdge. One line per change; each change ships with a test.

## Phase 2 — calibration & safety fixes (2026-11-14)
- fix: load_history normalizes BOM-mojibake column headers; validate flags NaN Div rows (silent multi-season data loss, AUDIT finding 0)
- fix: min_stake_pct threshold (default 0.5% of bankroll) — dust stakes are never placed or published (Phase 2 item 1)
- fix: leak-free per-league totals-bias correction on P(over 2.5) (AUDIT: unders over-predicted ~3.4pp) (Phase 2 item 4)

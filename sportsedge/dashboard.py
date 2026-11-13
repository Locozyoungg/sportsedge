"""Streamlit dashboard (Phase 4): bankroll curve, CLV, calibration, per-league ROI, open bets,
learning history. Run: streamlit run sportsedge/dashboard.py

Reads the paper ledger, reports/, and models/ — everything the pipeline already writes.
Degrades gracefully when the ledger is empty or the data dir is missing.
"""
from __future__ import annotations
import json
import os
import sys

# streamlit executes this file as a plain script (no package context), so relative imports
# would fail; make the project root importable and use absolute imports throughout.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml

from sportsedge.ledger import Ledger

LEARNING_LOG = "models/learning_log.jsonl"
REPORTS_DIR = "reports"


def _load_cfg(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    from sportsedge.learning import apply_champion
    return apply_champion(cfg)


def _read_learning_log() -> pd.DataFrame:
    if not os.path.exists(LEARNING_LOG):
        return pd.DataFrame()
    rows = [json.loads(l) for l in open(LEARNING_LOG, encoding="utf-8") if l.strip()]
    return pd.DataFrame(rows)


def _read_postmortems() -> list[tuple[str, str]]:
    out = []
    if os.path.isdir(REPORTS_DIR):
        for f in sorted(os.listdir(REPORTS_DIR), reverse=True):
            if f.startswith("postmortem_") and f.endswith(".md"):
                out.append((f, os.path.join(REPORTS_DIR, f)))
    return out


def drawdown_series(bets: pd.DataFrame, bankroll: float) -> pd.DataFrame:
    """Bankroll and drawdown series from settled bets, ordered by settlement time."""
    if bets.empty:
        return pd.DataFrame(columns=["ts", "bankroll", "drawdown"])
    b = bets.sort_values("ts").copy()
    b["bankroll"] = bankroll + b["pnl"].cumsum()
    peak = b["bankroll"].cummax()
    b["drawdown"] = (b["bankroll"] - peak) / peak
    return b[["ts", "bankroll", "drawdown"]]


def clv_summary(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty or "clv" not in bets.columns:
        return pd.DataFrame()
    c = bets.dropna(subset=["clv"])
    if c.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "n": [len(c)], "mean_clv": [c.clv.mean()], "clv_ci95_lo": [_boot_lo(c.clv)],
        "clv_ci95_hi": [_boot_hi(c.clv)], "share_positive": [(c.clv > 0).mean()],
    })


def _boot_lo(s, n: int = 2000, seed: int = 0):
    import numpy as np
    rng = np.random.default_rng(seed)
    a = np.asarray(s)
    return float(np.percentile([rng.choice(a, len(a)).mean() for _ in range(n)], 2.5))


def _boot_hi(s, n: int = 2000, seed: int = 0):
    import numpy as np
    rng = np.random.default_rng(seed)
    a = np.asarray(s)
    return float(np.percentile([rng.choice(a, len(a)).mean() for _ in range(n)], 97.5))


def per_league_roi(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()
    if "div" not in bets.columns:
        return pd.DataFrame()
    rows = []
    for div, g in bets.groupby("div"):
        rows.append({"league": div, "n": len(g), "staked": g.stake.sum(),
                     "pnl": g.pnl.sum(), "roi": g.pnl.sum() / g.stake.sum() if g.stake.sum() else 0.0})
    return pd.DataFrame(rows)


def open_bets_frame(ledger: Ledger) -> pd.DataFrame:
    rows = ledger.open_bets()
    return pd.DataFrame(rows, columns=["id", "match_date", "home", "away", "market",
                                       "selection", "line", "odds", "stake"])


def main() -> None:
    import streamlit as st
    from dotenv import load_dotenv
    load_dotenv()   # Telegram/odds credentials if the dashboard posts anything

    st.set_page_config(page_title="SportsEdge", page_icon="⚽", layout="wide")
    st.title("⚽ SportsEdge — paper trading dashboard")

    cfg_path = st.sidebar.text_input("config", "config.yaml")
    if not os.path.exists(cfg_path):
        st.error(f"config not found: {cfg_path}. Run from the project root.")
        st.stop()
    cfg = _load_cfg(cfg_path)
    bankroll = float(cfg["betting"]["bankroll"])

    ledger = Ledger(cfg["paths"]["ledger"])
    bets = ledger.settled_bets()
    if not bets.empty and "clv" not in bets.columns:
        bets["clv"] = None
    preds = ledger.settled_predictions()

    stop = os.path.exists("STOP")
    if stop:
        st.error("STOP file present — autopilot is PAUSED. Read STOP, fix the cause, then delete it.")

    c1, c2, c3, c4, c5 = st.columns(5)
    n, staked, pnl = ledger.report()
    c1.metric("Settled bets", f"{n}")
    c2.metric("P&L (paper)", f"{pnl:+.2f}", f"ROI {pnl / staked:.1%}" if staked else "no bets")
    c3.metric("Open bets", f"{len(open_bets_frame(ledger))}")
    c4.metric("Settled predictions", f"{len(preds)}")
    c5.metric("Version", cfg.get("_version", "v0"))

    tab_overview, tab_clv, tab_cal, tab_leagues, tab_open, tab_predictions, tab_learning, tab_backtest, tab_reports = \
        st.tabs(["Overview", "CLV", "Calibration", "Leagues", "Open bets",
                 "Predictions", "Learning", "Backtest", "Reports"])

    with tab_overview:
        st.subheader("Bankroll & drawdown")
        if bets.empty:
            st.info("No settled bets yet. Run `python -m sportsedge paper` to start the paper record.")
        else:
            dd = drawdown_series(bets, bankroll)
            st.line_chart(dd.set_index("ts")["bankroll"], height=300)
            st.line_chart(dd.set_index("ts")["drawdown"], height=200)
            st.caption("Drawdown is share below the running peak bankroll.")
            mdd = dd.drawdown.min()
            st.metric("Max drawdown", f"{mdd:.1%}")
        st.subheader("Bet log")
        st.dataframe(bets.sort_values("match_date", ascending=False)
                     if not bets.empty else pd.DataFrame(), use_container_width=True)

    with tab_clv:
        st.subheader("Closing-line value")
        c = clv_summary(bets)
        if c.empty:
            st.info("CLV is recorded per 1X2 bet at settlement time, once closing odds are in the data.")
        else:
            r = c.iloc[0]
            st.metric("Mean CLV", f"{r.mean_clv:.2%}", help="Persistently > 0 is the real sign of skill")
            st.metric("CLV 95% CI", f"[{r.clv_ci95_lo:.2%}, {r.clv_ci95_hi:.2%}]")
            st.metric("Share positive", f"{r.share_positive:.0%}")
            st.dataframe(bets.dropna(subset=["clv"])[["match_date", "home", "away", "selection", "clv"]]
                         .sort_values("match_date", ascending=False), use_container_width=True)

    with tab_cal:
        st.subheader("Calibration (from settled predictions)")
        if preds.empty:
            st.info("No settled predictions yet — the ledger fills as `paper`/`autopilot` runs.")
        else:
            from sportsedge.postmortem import calibration_bins, IDX
            pf = preds[["pf_h", "pf_d", "pf_a"]].to_numpy()
            y = preds.outcome.map(IDX).to_numpy()
            bins = calibration_bins(pf, y)
            st.dataframe(pd.DataFrame(bins), use_container_width=True)
            st.caption("Predicted ≈ actual in every bin means the probabilities are honest.")

    with tab_leagues:
        st.subheader("ROI per league")
        st.dataframe(per_league_roi(bets), use_container_width=True)

    with tab_open:
        st.subheader("Open bets")
        st.dataframe(open_bets_frame(ledger), use_container_width=True)

    with tab_predictions:
        st.subheader("Settled predictions")
        st.dataframe(preds, use_container_width=True)

    with tab_learning:
        st.subheader("Learning history (models/learning_log.jsonl)")
        log = _read_learning_log()
        if log.empty:
            st.info("No learning runs yet. `python -m sportsedge learn` runs the gated improvement step.")
        else:
            st.dataframe(log.sort_values("ts", ascending=False), use_container_width=True)
        champ_path = os.path.join(cfg["learning"]["models_dir"], "champion.json")
        if os.path.exists(champ_path):
            with open(champ_path) as f:
                st.json(json.load(f))

    with tab_backtest:
        st.subheader("Run a backtest")
        st.caption("Walk-forward, per-league. Uses only data before each prediction. May take minutes.")
        raw = cfg["paths"]["raw_dir"]
        if not os.path.isdir(raw):
            st.warning("No data directory. Run `python -m sportsedge fetch` first.")
        elif st.button("Run backtest", type="primary"):
            from sportsedge import data
            from sportsedge.backtest import run_backtest, summarize
            hist = data.load_history(raw)
            prog = st.progress(0.0, text="running...")
            for i, (div, g) in enumerate(hist.groupby("Div")):
                st.markdown(f"### {div}")
                st.code(summarize(run_backtest(g, cfg)))
                prog.progress((i + 1) / hist.Div.nunique())
            prog.empty()

    with tab_reports:
        st.subheader("Post-mortem reports (reports/)")
        pm = _read_postmortems()
        if not pm:
            st.info("No post-mortems yet. They are written by `autopilot` / `postmortem`.")
        for name, path in pm[:14]:
            with st.expander(name):
                st.code(open(path, encoding="utf-8").read(), language="markdown")

    st.caption("Paper mode. Live execution is disabled until the Live Gates pass and you approve it.")


if __name__ == "__main__":
    main()

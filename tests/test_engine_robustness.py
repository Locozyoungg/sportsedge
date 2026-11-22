"""Engine robustness tests (prompt Phase 2, engine only): loader, validation, leakage, determinism, sanity guards."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from sportsedge import data


# ---------- A1: single robust loader ----------

def _write(path, text, encoding="utf-8"):
    path.write_text(text, encoding=encoding)


def test_loader_mixed_headers_dedupes_and_parses_both_date_formats(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw / "2324_D1.csv",  # clean header, dd/mm/yyyy
           "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nD1,23/08/2024,A,B,1,0,H\nD1,24/08/2024,C,D,2,2,D\n")
    _write(raw / "2425_D1.csv",  # mojibake BOM header, dd/mm/yy, one duplicate row from the first file
           "ï»¿Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nD1,23/08/25,E,F,0,3,A\nD1,24/08/24,C,D,2,2,D\n")
    _write(raw / "2526_D1.csv",  # lowercase headers
           "div,date,hometeam,awayteam,fthg,ftag,ftr\nD1,23/08/2025,G,H,1,1,D\n")
    df = data.load_history(str(raw))
    assert df.Div.notna().all() and set(df.Div) == {"D1"}
    assert len(df) == 4, f"expected 4 rows (5 minus 1 true duplicate), got {len(df)}"
    assert not df.duplicated(["Div", "Date", "HomeTeam", "AwayTeam"]).any()
    # both date formats parsed: 2-digit-year row exists with year 2025
    assert (df.Date.dt.year == 2025).sum() == 2


def test_loader_raises_loudly_on_unparseable_dates(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw / "2324_D1.csv",
           "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nD1,2024-08-23,A,B,1,0,H\nD1,not-a-date,C,D,2,2,D\n")
    with pytest.raises(ValueError, match="unparseable dates"):
        data.load_history(str(raw))


def test_loader_raises_loudly_on_missing_required_columns(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw / "2324_D1.csv", "Div,Date,HomeTeam,FTHG,FTAG,FTR\nD1,23/08/2024,A,1,0,H\n")
    with pytest.raises(ValueError, match="missing required columns"):
        data.load_history(str(raw))


def test_loader_utf8_bom_and_latin1_mojibake_bytes(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "2324_D1.csv").write_bytes(
        b"\xef\xbb\xbfDiv,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nD1,23/08/2024,A,B,1,0,H\n")
    # a real pre-fix mojibake file: the utf-8 BOM bytes were latin-1-decoded then re-encoded as utf-8
    mojibake = "ï»¿Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nD1,23/08/2025,C,D,2,2,D\n"
    (raw / "2425_D1.csv").write_bytes(mojibake.encode("utf-8"))
    df = data.load_history(str(raw))
    assert len(df) == 2 and df.Div.notna().all()


def test_fixtures_lowercase_header_via_single_normalizer(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw / "fixtures.csv", "div,Date,HomeTeam,AwayTeam\nE0,23/08/2026,Arsenal,Chelsea\n")
    fx = data.load_fixtures(str(raw))
    assert list(fx.columns)[:4] == ["Div", "Date", "HomeTeam", "AwayTeam"]
    assert fx.Div.iloc[0] == "E0"


# ---------- A2: strong validation ----------

def _vcfg(**over):
    c = {"betting": {"venues": {"bookmaker": {"cols": ["B365H", "B365D", "B365A"]}}},
         "validate": {"min_rows": 2, "max_goals": 15}}
    c["validate"].update(over)
    return c


def _frame(rows):
    d = pd.DataFrame(rows, columns=["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                                    "B365H", "B365D", "B365A"])
    d["Date"] = pd.to_datetime(d["Date"])
    return d


def _clean():
    return _frame([
        ["E0", "2025-08-23", "A", "B", 1, 0, "H", 1.8, 3.4, 4.5],
        ["E0", "2025-08-24", "C", "D", 2, 2, "D", 2.6, 3.2, 2.8],
        ["E0", "2026-03-14", "A", "C", 0, 3, "A", 4.0, 3.5, 1.9],
    ])


def test_validate_passes_clean_frame():
    from sportsedge.validate import validate_history
    rep = validate_history(_clean(), _vcfg())
    assert rep["ok"], rep["fatal"]


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.assign(Date=d.Date.mask([True] + [False] * 2)), "missing Date"),
    (lambda d: d.assign(HomeTeam=lambda x: x.HomeTeam.mask([False, True, False], "")), "missing HomeTeam"),
    (lambda d: d.assign(FTHG=[1, 2, 99]), "implausible score"),
    (lambda d: d.assign(Date=pd.to_datetime(["2025-08-23", "2025-08-24", "2031-03-14"])), "future"),
    (lambda d: d.iloc[:2], "below validate.min_rows"),
])
def test_validate_flags_new_fatals(mutate, needle):
    from sportsedge.validate import validate_history
    rep = validate_history(mutate(_clean()), _vcfg(min_rows=3))
    assert not rep["ok"] and any(needle in f for f in rep["fatal"]), rep


def test_assert_valid_aborts_engine_commands():
    from sportsedge.cli import _assert_valid
    from sportsedge.validate import validate_history
    bad = _clean().assign(FTHG=[1, 2, 99])
    with pytest.raises(SystemExit, match="implausible score"):
        _assert_valid(_vcfg(), bad)


# ---------- A3: team-name consistency ----------

def test_team_aliases_file_canonicalises_history_and_is_shared_by_settle(tmp_path):
    raw, parent = tmp_path / "raw", tmp_path
    raw.mkdir()
    _write(raw / "2324_E0.csv",
           "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nE0,2024-08-23,Man United,Liverpool,1,0,H\n")
    _write(raw / "2425_E0.csv",
           "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nE0,2025-08-23,Manchester United,Liverpool,2,2,D\n")
    _write(parent / "team_aliases.csv", "name,canonical\nManchester United,Man United\n")
    df = data.load_history(str(raw))
    names = set(df.HomeTeam) | set(df.AwayTeam)
    assert names == {"Man United", "Liverpool"}, names
    # the settle path uses the same canonicalisation for ledger names
    assert data.canonical_team("Manchester United", data._aliases_for(str(raw))) == "Man United"


def test_team_aliases_missing_or_empty_file_is_a_noop(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw / "2324_E0.csv",
           "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nE0,2024-08-23,A,B,1,0,H\n")
    _write(tmp_path / "team_aliases.csv", "# comment\nname,canonical\n")
    assert data.load_team_aliases(str(tmp_path / "team_aliases.csv")) == {}
    df = data.load_history(str(raw))
    assert set(df.HomeTeam) == {"A"}


# ---------- A4: leakage and determinism ----------

def _row(h, a, gh, ga, d):
    return {"HomeTeam": h, "AwayTeam": a, "FTHG": gh, "FTAG": ga, "HS": 10, "HST": 5, "AS": 4, "AST": 1,
            "Date": pd.Timestamp(d)}


def _feat_equal(d1, d2) -> bool:
    """dict equality that treats NaN == NaN (feature dicts are full of NaNs)."""
    return d1.keys() == d2.keys() and all(
        (v1 == v2) or (isinstance(v1, float) and isinstance(v2, float) and np.isnan(v1) and np.isnan(v2))
        for (k1, v1), (k2, v2) in zip(sorted(d1.items()), sorted(d2.items())))


def test_feature_state_never_uses_future_matches():
    from sportsedge.features import FeatureState
    st = FeatureState()
    st.update(_row("A", "B", 2, 1, "2025-08-10"))
    before = st.features("C", "D", pd.Timestamp("2025-08-17"))
    st.update(_row("C", "D", 0, 3, "2025-08-24"))  # a match AFTER the query date
    after = st.features("C", "D", pd.Timestamp("2025-08-17"))
    assert _feat_equal(before, after)
    # and the same guard holds when updates arrive out of order
    st2 = FeatureState()
    st2.update(_row("C", "D", 0, 3, "2025-08-24"))  # future-dated update first
    assert _feat_equal(st2.features("C", "D", pd.Timestamp("2025-08-17")),
                       {"elo_h": 1500.0, "elo_a": 1500.0, "elo_diff": 65.0,
                        "form_pts_diff": float("nan"), "form_gd_diff": float("nan"),
                        "xg_diff": float("nan"), "rest_h": float("nan"), "rest_a": float("nan"),
                        "rest_diff": float("nan"), "cong_h": 0.0, "cong_a": 0.0, "inj_h": 0.0, "inj_a": 0.0,
                        "overround": float("nan"), "mkt_h": float("nan"), "mkt_d": float("nan"),
                        "mkt_a": float("nan"), "mkt_lr": float("nan")})


def test_poisson_oos_predictions_stable_when_future_appended():
    from sportsedge.ensemble import poisson_oos
    from tests.test_core import synth
    import yaml, pathlib
    cfg = yaml.safe_load(open(pathlib.Path(__file__).parents[1] / "config.yaml"))
    cfg["backtest"]["min_train_matches"] = 150
    df = synth(seasons=3)
    base = poisson_oos(df, cfg)
    more = synth(seasons=3)
    more["Date"] += pd.Timedelta(days=365 * 3)                 # strictly after the original data
    more.index = range(len(df), len(df) + len(more))           # keep original rids stable across the concat
    extended = poisson_oos(pd.concat([df, more]).sort_values("Date"), cfg)
    assert len(base) > 0
    for rid, p in base.items():
        assert rid in extended and np.allclose(p, extended[rid])


def test_fit_as_of_excludes_future_teams():
    from sportsedge.model import PoissonModel
    from tests.test_core import synth
    df = synth(seasons=2)
    mid = df.Date.min() + (df.Date.max() - df.Date.min()) / 2
    later_only = pd.DataFrame([_row("X", "A", 1, 0, (df.Date.max() + pd.Timedelta(days=1)).isoformat())])
    m = PoissonModel().fit(pd.concat([df, later_only], ignore_index=True), as_of=mid)
    assert not m.knows("X", "A")          # X only played after as_of: must not be known
    # with the new-team prior the fallback still predicts (league-average strengths)
    m2 = PoissonModel(newteam_prior=True).fit(pd.concat([df, later_only], ignore_index=True), as_of=mid)
    p = m2.predict_1x2("X", "A")
    assert np.all(np.isfinite(p)) and abs(p.sum() - 1) < 1e-9


def test_backtest_is_deterministic_given_same_seed():
    from sportsedge.backtest import run_backtest
    from tests.test_core import synth
    import yaml, pathlib
    cfg = yaml.safe_load(open(pathlib.Path(__file__).parents[1] / "config.yaml"))
    cfg["backtest"]["min_train_matches"] = 150
    cfg["stack"]["min_train"] = 120
    df = synth(seasons=2)
    r1, r2 = run_backtest(df, cfg), run_backtest(df, cfg)
    pd.testing.assert_frame_equal(r1.pop("bets"), r2.pop("bets"))
    assert r1 == r2, {k: (r1.get(k), r2.get(k)) for k in r1.keys() | r2.keys() if r1.get(k) != r2.get(k)}


# ---------- A5: probability sanity guards ----------

def _grid11(lh=1.6, la=1.1):
    from scipy.stats import poisson
    k = np.arange(11)
    g = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
    return g / g.sum()


def test_dc_correction_never_produces_negative_cells():
    from sportsedge.model import PoissonModel
    m = PoissonModel(dc=True)
    m.rho = 0.3                      # max allowed; at lambda 2 x 2, 1 - lh*la*rho goes negative
    m._lambdas = lambda h, a: (2.0, 2.0)
    g = m.score_grid("X", "Y")
    assert (g >= 0).all() and np.all(np.isfinite(g)) and abs(g.sum() - 1) < 1e-9


def test_score_grid_raises_when_tail_mass_exceeds_limit():
    from sportsedge.model import PoissonModel
    m = PoissonModel(max_goals=2, max_tail_mass=0.01)
    m._lambdas = lambda h, a: (6.0, 6.0)
    with pytest.raises(ValueError, match="tail mass"):
        m.score_grid("X", "Y")


def test_sane_probs_clips_and_renormalises_garbage(capsys):
    from sportsedge.strategy import _sane_probs, CLIP_LO
    p = _sane_probs(np.array([0.9, np.nan, -0.1]), "test ctx")
    assert np.all(np.isfinite(p)) and (p >= CLIP_LO).all() and abs(p.sum() - 1) < 1e-9
    assert "WARNING probabilities[test ctx]" in capsys.readouterr().out
    # sane input passes through untouched and warns nothing
    q = _sane_probs(np.array([0.5, 0.3, 0.2]), "ok")
    assert np.allclose(q, [0.5, 0.3, 0.2])
    assert "ok" not in capsys.readouterr().out


def test_strategy_probabilities_guard_end_to_end(capsys):
    from sportsedge.strategy import probabilities
    import yaml, pathlib
    cfg = yaml.safe_load(open(pathlib.Path(__file__).parents[1] / "config.yaml"))

    class Absurd:
        kind = "poisson"
        def knows(self, row): return True
        def one_x_two(self, row): return np.array([0.8, np.nan, 0.4])   # NaN + sums > 1
        def grid(self, row): return _grid11()
        def ou_prob(self, row): return 0.5

    row = pd.Series({"HomeTeam": "A", "AwayTeam": "B", "B365H": 2.0, "B365D": 3.2, "B365A": 4.0,
                     "PSH": 1.9, "PSD": 3.3, "PSA": 4.2, "B365>2.5": 1.85, "B365<2.5": 1.95,
                     "P>2.5": 1.8, "P<2.5": 1.9})
    pr = probabilities(Absurd(), row, cfg)
    f = pr["x2"]["final"]
    assert np.all(np.isfinite(f)) and (f >= 1e-4).all() and abs(f.sum() - 1) < 1e-3
    assert "WARNING probabilities[model 1X2]" in capsys.readouterr().out
    assert 1e-4 <= pr["ou"]["final"] <= 1 - 1e-4


# ---------- A6: backtest metrics ----------

def test_backtest_reports_brier_and_calibration_bins():
    from sportsedge.backtest import run_backtest
    from tests.test_core import synth
    import yaml, pathlib
    cfg = yaml.safe_load(open(pathlib.Path(__file__).parents[1] / "config.yaml"))
    cfg["backtest"]["min_train_matches"] = 150
    cfg["stack"]["min_train"] = 120
    r = run_backtest(synth(seasons=2), cfg)
    for k in ("brier_model", "brier_market", "brier_blend"):
        assert 0 <= r[k] <= 2, (k, r[k])
    bins = r["calibration_bins"]
    assert len(bins) == 10 and sum(b["n"] for b in bins) == r["n_predictions"]
    assert all(0 <= b["conf"] <= 1 and 0 <= b["acc"] <= 1 for b in bins if b["n"])


# ---------- B5: injury adjustment (config-gated) ----------

def test_injuries_adjust_poisson_path_only_when_enabled():
    from sportsedge.ensemble import PoissonPredictor
    from sportsedge.model import PoissonModel
    from tests.test_core import synth

    class Inj:
        def impact(self, team, date):
            return 2.0 if team == "T1" else 0.0

    df = synth(seasons=2)
    m = PoissonModel().fit(df)
    row = pd.Series({"HomeTeam": "T1", "AwayTeam": "T2", "Date": df.Date.iloc[-1]})
    off = PoissonPredictor(m, ext=Inj(), use_injuries=False)
    on = PoissonPredictor(m, ext=Inj(), injury_strength=0.15, use_injuries=True)
    p_off, p_on = off.one_x_two(row), on.one_x_two(row)
    assert np.allclose(p_off, m.predict_1x2("T1", "T2"))   # flag off = unchanged engine
    assert abs(p_on.sum() - 1) < 1e-9 and p_on[0] < p_off[0]  # weakened home attack -> lower H
    g = on.grid(row)
    assert (g >= 0).all() and abs(g.sum() - 1) < 1e-9

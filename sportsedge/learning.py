"""Autonomous improvement loop with strict gates.
The agent may change ONLY: half_life_days, reg, market_weight, calibration. Never risk limits, staking, live flags.
A challenger is promoted only if it beats the champion on a HELD-OUT, later period with a paired-bootstrap CI."""
from __future__ import annotations
import datetime as dt, json, os
import numpy as np
import pandas as pd
from .calibration import Calibrator
from .model import PoissonModel
from .odds import devig_power
from .odds import get_odds
from .ensemble import build_table, stack_walk_forward

IDX = {"H": 0, "D": 1, "A": 2}


# ---------- champion registry ----------
def _paths(cfg):
    d = cfg["learning"]["models_dir"]
    return d, os.path.join(d, "champion.json"), os.path.join(d, "versions"), os.path.join(d, "learning_log.jsonl")


def load_champion(cfg):
    _, champ, _, _ = _paths(cfg)
    if os.path.exists(champ):
        with open(champ) as f:
            return json.load(f)
    return None


def apply_champion(cfg):
    c = load_champion(cfg)
    if c:
        cfg["model"].update(c["model"])
        cfg["_calibrator"] = Calibrator.from_dict(c["calibration"])
        cfg["_version"] = f"v{c['version']}"
    else:
        cfg["_version"] = "v0"
    return cfg


def _log(cfg, entry):
    d, _, _, logp = _paths(cfg)
    os.makedirs(d, exist_ok=True)
    with open(logp, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def last_log(cfg, kind=None):
    _, _, _, logp = _paths(cfg)
    if not os.path.exists(logp):
        return None
    rows = [json.loads(l) for l in open(logp) if l.strip()]
    if kind:
        rows = [r for r in rows if r.get("decision") == kind]
    return rows[-1] if rows else None


def save_version(cfg, model_params, calibrator, evidence):
    d, champ, vdir, _ = _paths(cfg)
    os.makedirs(vdir, exist_ok=True)
    prev = load_champion(cfg)
    n = (prev["version"] if prev else 0) + 1
    rec = {"version": n, "created": dt.datetime.now(dt.UTC).replace(tzinfo=None).isoformat(), "model": model_params,
           "calibration": calibrator.to_dict(), "evidence": evidence}
    for p in (os.path.join(vdir, f"v{n}.json"), champ):
        with open(p, "w") as f:
            json.dump(rec, f, indent=2, default=str)
    return n


def rollback(cfg, version: int):
    d, champ, vdir, _ = _paths(cfg)
    src = os.path.join(vdir, f"v{version}.json")
    if not os.path.exists(src):
        raise SystemExit(f"no such version {version}")
    with open(src) as f, open(champ, "w") as g:
        g.write(f.read())
    _log(cfg, {"ts": dt.datetime.now(dt.UTC).replace(tzinfo=None).isoformat(), "decision": "rollback", "to": version})


# ---------- walk-forward probabilities ----------
def walk_forward(df: pd.DataFrame, hl: float, reg: float, cfg: dict) -> dict:
    mc, bc, kc, L = cfg["model"], cfg["betting"], cfg["backtest"], cfg["learning"]
    rid, dates, y, PM, PK = [], [], [], [], []
    for _, g in df.groupby("Div"):
        g = g.sort_values("Date")
        if len(g) <= kc["min_train_matches"]:
            continue
        cur, last = g.Date.iloc[kc["min_train_matches"]], g.Date.max()
        while cur <= last:
            end = cur + pd.Timedelta(days=L["refit_days"])
            win = g[(g.Date >= cur) & (g.Date < end)]
            if len(win):
                m = PoissonModel(hl, reg, mc["max_goals"], mc.get("dc", True), mc.get("newteam_prior", False)).fit(g, as_of=cur)
                for i, row in win.iterrows():
                    if not m.knows(row.HomeTeam, row.AwayTeam):
                        continue
                    bo = get_odds(row, bc["venues"]["bookmaker"]["cols"])
                    if bo is None:
                        continue
                    sh = get_odds(row, bc["sharp_cols"])
                    rid.append(i); dates.append(row.Date); y.append(IDX[row.FTR])
                    PM.append(m.predict_1x2(row.HomeTeam, row.AwayTeam))
                    PK.append(devig_power(sh if sh is not None else bo))
            cur = end
    o = np.argsort(rid, kind="stable")
    return {"rid": np.array(rid)[o], "date": np.array(dates)[o], "y": np.array(y)[o],
            "PM": np.array(PM)[o], "PK": np.array(PK)[o]}


def blend_ll(r: dict, w: float, cal: Calibrator) -> np.ndarray:
    final = w * r["PK"] + (1 - w) * cal.apply(r["PM"])
    return -np.log(np.clip(final[np.arange(len(r["y"])), r["y"]], 1e-9, 1))


def _boot_ci(diff, ci, seed=0, n=2000):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(diff, len(diff)).mean() for _ in range(n)])
    return float(np.percentile(means, 100 * (1 - ci))), float(np.percentile(means, 100 * ci))


# ---------- the improvement step ----------
def improve(cfg: dict, df: pd.DataFrame) -> dict:
    """Champion/challenger over two model families: 'poisson' (Dixon-Coles + calibration + market blend) and
    'stack' (ensemble). Tuned on an earlier window, judged once on a later held-out window."""
    L = cfg["learning"]
    df = df.sort_values("Date").reset_index(drop=True)
    champ_type = cfg["model"].get("type", "poisson")
    champ_m = {k: cfg["model"][k] for k in ("half_life_days", "reg", "market_weight")}
    champ_m["type"] = champ_type
    champ_cal = cfg.get("_calibrator") or Calibrator()
    cache = {}
    ps = None
    if L.get("try_stack", True) or champ_type == "stack":
        table, _ = build_table(df, cfg)
        ps = stack_walk_forward(table, cfg)
    base0 = walk_forward(df, champ_m["half_life_days"], champ_m["reg"], cfg)
    common = np.intersect1d(base0["rid"], ps.index.to_numpy()) if ps is not None and len(ps) else None

    def restrict(r):
        if common is None:
            return r
        m = np.isin(r["rid"], common)
        return {k: v[m] for k, v in r.items()}

    def wf(hl, reg):
        k = (hl, reg)
        if k not in cache:
            cache[k] = restrict(base0 if k == (champ_m["half_life_days"], champ_m["reg"]) else walk_forward(df, hl, reg, cfg))
        return cache[k]

    base = wf(champ_m["half_life_days"], champ_m["reg"])
    n = len(base["y"])
    decision = {"ts": dt.datetime.now(dt.UTC).replace(tzinfo=None).isoformat(), "n_predictions": n, "champion": champ_m,
                "champion_cal": champ_cal.to_dict()}
    cut = int(n * (1 - L["holdout_frac"]))
    if n - cut < L["min_holdout_matches"]:
        decision.update(decision="kept", reason=f"holdout too small ({n - cut} < {L['min_holdout_matches']})")
        _log(cfg, decision)
        return decision

    stack_r = None
    if common is not None and len(common):
        stack_r = {**base, "PM": ps.loc[base["rid"], ["p_h", "p_d", "p_a"]].to_numpy()}
    ident = Calibrator()

    g = L["grid"]
    best = None                                   # (tune_ll, type, hl, reg, w, cal)
    for hl in g["half_life_days"]:
        for reg in g["reg"]:
            r = wf(hl, reg)
            cal = Calibrator.fit(r["PM"][:cut], r["y"][:cut]) if L["calibrate"] else Calibrator()
            for w in g["market_weight"]:
                ll = blend_ll(r, w, cal)[:cut].mean()          # chosen on TUNING data only
                if best is None or ll < best[0]:
                    best = (ll, "poisson", hl, reg, w, cal)
    if stack_r is not None:
        for w in g["market_weight"]:
            ll = blend_ll(stack_r, w, ident)[:cut].mean()
            if ll < best[0]:
                best = (ll, "stack", champ_m["half_life_days"], champ_m["reg"], w, ident)
    _, ctype, hl, reg, w, cal = best
    chall = {"half_life_days": hl, "reg": reg, "market_weight": w, "type": ctype}
    r_ch = stack_r if ctype == "stack" else wf(hl, reg)
    r_ch_cal = ident if ctype == "stack" else cal
    r_champ = stack_r if (champ_type == "stack" and stack_r is not None) else base
    champ_hold = blend_ll(r_champ, champ_m["market_weight"], ident if champ_type == "stack" else champ_cal)[cut:]
    chall_hold = blend_ll(r_ch, w, r_ch_cal)[cut:]
    market_hold = -np.log(np.clip(base["PK"][cut:][np.arange(n - cut), base["y"][cut:]], 1e-9, 1))
    diff = champ_hold - chall_hold                                # >0 means challenger better
    lo, hi = _boot_ci(diff, L["ci"], seed=int(L.get("seed", 0)))
    decision.update(challenger=chall, challenger_cal=r_ch_cal.to_dict(), holdout_n=n - cut,
                    ll_champion=float(champ_hold.mean()), ll_challenger=float(chall_hold.mean()),
                    ll_market=float(market_hold.mean()), gain=float(diff.mean()), gain_ci=[lo, hi])
    last = last_log(cfg, "promoted")
    recent = last and (dt.datetime.now(dt.UTC).replace(tzinfo=None) - dt.datetime.fromisoformat(last["ts"])).days < L["min_days_between_promotions"]
    same = (chall == champ_m and np.allclose(r_ch_cal.b, champ_cal.b, atol=1e-3) and abs(r_ch_cal.T - champ_cal.T) < 1e-3)
    if same:
        decision.update(decision="kept", reason="challenger identical to champion")
    elif recent:
        decision.update(decision="kept", reason="promotion cooldown (avoid chasing noise)")
    elif diff.mean() >= L["min_gain"] and lo > 0:
        v = save_version(cfg, chall, r_ch_cal, {k: decision[k] for k in ("holdout_n", "ll_champion", "ll_challenger", "ll_market", "gain", "gain_ci")})
        decision.update(decision="promoted", version=v, reason=f"{ctype} beat champion on held-out period with CI above 0")
    else:
        decision.update(decision="kept", reason="no statistically reliable improvement on held-out period")
    _log(cfg, decision)
    return decision

"""Polymarket integration: read-only market scan (public Gamma + CLOB APIs), price overlay for the strategy,
paper execution via the ledger, and a heavily gated LIVE broker.

IMPORTANT
- Check Polymarket's geographic restrictions for YOUR location yourself (they change, and sources disagree about Kenya).
  `polymarket-geocheck` calls their geoblock endpoint; live trading fails CLOSED if it says blocked or errors.
  Never use a VPN to get around restrictions: it breaches their terms.
- A YES share at ask c pays 1 if it wins, so decimal odds = 1/c and the same edge/Kelly maths applies.
- The live adapter is untested against the real exchange: start with the smallest order and manual approval."""
from __future__ import annotations
import json, os, re
import numpy as np
import pandas as pd
import requests
from . import names
from .broker import Broker


def _jl(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except ValueError:
            return []
    return x or []


class PolymarketClient:
    def __init__(self, cfg: dict, session=None):
        self.c = cfg["polymarket"]
        self.s = session or requests

    def _get(self, url, params=None):
        r = self.s.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def events(self, pages: int = 3, limit: int = 100) -> list:
        out = []
        for i in range(pages):
            batch = self._get(f"{self.c['gamma']}/events", {"active": "true", "closed": "false",
                              "tag_slug": self.c["tag_slug"], "limit": limit, "offset": i * limit})
            out += batch
            if len(batch) < limit:
                break
        return out

    def book(self, token_id: str) -> dict:
        return self._get(f"{self.c['clob']}/book", {"token_id": token_id})


def parse_market(m: dict) -> dict:
    return {"question": m.get("question") or "", "outcomes": [str(o) for o in _jl(m.get("outcomes"))],
            "prices": [float(p) for p in _jl(m.get("outcomePrices"))], "tokens": [str(t) for t in _jl(m.get("clobTokenIds"))],
            "liquidity": float(m.get("liquidity") or 0), "end": m.get("gameStartTime") or m.get("endDate"),
            "condition": m.get("conditionId"), "slug": m.get("slug")}


def _forms(team: str) -> set:
    n = names.normalize(team)
    f = {names._clean(team), n}
    f |= {k for k, v in names.ALIASES.items() if v == n}
    return {x for x in f if len(x) > 2}


def _first_pos(text: str, team: str) -> int:
    pos = [text.find(f) for f in _forms(team) if re.search(rf"\b{re.escape(f)}\b", text)]
    return min(pos) if pos else -1


def classify(market: dict, home: str, away: str):
    """Selection ('H','D','A') for a market, or None. Handles Yes/No markets and two-team moneylines."""
    outs = [o.lower() for o in market["outcomes"]]
    q = names._clean(market["question"])
    if outs == ["yes", "no"]:
        if "draw" in q:
            return "D"
        ph, pa = _first_pos(q, home), _first_pos(q, away)
        if ph < 0 and pa < 0:
            return None
        if ph >= 0 and (pa < 0 or ph < pa):
            return "H"
        return "A"
    return None


def parse_book(book: dict):
    bids = [(float(b["price"]), float(b["size"])) for b in book.get("bids", [])]
    asks = [(float(a["price"]), float(a["size"])) for a in book.get("asks", [])]
    return (max(bids)[0] if bids else None), (min(asks)[0] if asks else None), asks


def scan(fx: pd.DataFrame, events: list, client: PolymarketClient, cfg: dict) -> pd.DataFrame:
    """Add PM_H/D/A (decimal odds = 1/ask), PM_MID_*, and PM_TOKEN_* columns to fixtures."""
    pc, fx = cfg["polymarket"], fx.copy()
    for c in ("PM_H", "PM_D", "PM_A", "PM_MID_H", "PM_MID_D", "PM_MID_A"):
        fx[c] = np.nan
    for c in ("PM_TOKEN_H", "PM_TOKEN_D", "PM_TOKEN_A"):
        fx[c] = None
    for i, r in fx.iterrows():
        for ev in events:
            title = names._clean(ev.get("title", ""))
            if _first_pos(title, r.HomeTeam) < 0 or _first_pos(title, r.AwayTeam) < 0:
                continue
            for m in ev.get("markets", []):
                pm = parse_market(m)
                sel = classify(pm, r.HomeTeam, r.AwayTeam)
                if sel is None or len(pm["tokens"]) < 1 or pm["liquidity"] < pc["min_liquidity_usd"]:
                    continue
                if pm["end"]:
                    try:
                        if abs((pd.Timestamp(pm["end"]).tz_localize(None) - r.Date).days) > 3:
                            continue
                    except (ValueError, TypeError):
                        pass
                try:
                    bid, ask, _ = parse_book(client.book(pm["tokens"][0]))
                except Exception as e:  # network/schema problem: skip, never guess a price
                    print("book error:", e)
                    continue
                if ask is None or bid is None or ask - bid > pc["max_spread"] or not (0 < ask < 1):
                    continue
                fx.at[i, f"PM_{sel}"] = 1.0 / ask
                fx.at[i, f"PM_MID_{sel}"] = (ask + bid) / 2
                fx.at[i, f"PM_TOKEN_{sel}"] = pm["tokens"][0]
            break
    return fx


def check_geoblock(cfg: dict, session=None):
    """(allowed, info). FAILS CLOSED on any error."""
    try:
        r = (session or requests).get(cfg["polymarket"]["geoblock"], timeout=15)
        r.raise_for_status()
        d = r.json()
        return (d.get("blocked") is False), d
    except Exception as e:
        return False, {"error": str(e)}


class PolymarketLiveBroker(Broker):
    """Places LIMIT BUY orders for YES shares via py-clob-client (pip install py-clob-client).
    Gates: config polymarket.live.enabled, LIVE_CONFIRM=I_ACCEPT_RISK, wallet key present, geoblock allows, no STOP file,
    per-order USD cap, manual approval. Use a dedicated wallet holding only what you can afford to lose."""

    def __init__(self, cfg: dict, ledger, session=None):
        pc = cfg["polymarket"]
        if not pc.get("live", {}).get("enabled"):
            raise RuntimeError("polymarket.live.enabled is false")
        if os.getenv("LIVE_CONFIRM") != "I_ACCEPT_RISK":
            raise RuntimeError("Set LIVE_CONFIRM=I_ACCEPT_RISK to arm live mode")
        if not os.getenv("POLYMARKET_PRIVATE_KEY"):
            raise RuntimeError("POLYMARKET_PRIVATE_KEY not set")
        ok, info = check_geoblock(cfg, session)
        if not ok:
            raise RuntimeError(f"geoblock check failed or blocked: {info}")
        self.cfg, self.ledger = cfg, ledger
        self.manual = cfg["live"].get("require_manual_approval", True)
        self._client = None

    def _clob(self):
        if self._client is None:
            from py_clob_client.client import ClobClient  # lazy: optional dependency
            c = ClobClient(self.cfg["polymarket"]["clob"], key=os.environ["POLYMARKET_PRIVATE_KEY"], chain_id=137,
                           signature_type=int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0")),
                           funder=os.getenv("POLYMARKET_FUNDER") or None)
            c.set_api_creds(c.create_or_derive_api_creds())
            self._client = c
        return self._client

    def place(self, bet: dict) -> bool:
        if os.path.exists("STOP"):
            return False
        usd = min(bet["stake"], self.cfg["polymarket"]["max_order_usd"])
        price = round(1.0 / bet["odds"], 2)           # YES ask; 2dp matches the usual tick size, verify per market
        if not bet.get("ref") or usd < 1 or not (0.02 < price < 0.98):
            return False
        shares = round(usd / price, 2)
        if self.manual and input(f"BUY {shares} YES @ {price} (${usd}) on {bet['home']} v {bet['away']} [{bet['selection']}]? type YES: ") != "YES":
            return False
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY
        c = self._clob()
        resp = c.post_order(c.create_order(OrderArgs(token_id=bet["ref"], price=price, size=shares, side=BUY)), OrderType.GTC)
        if not (isinstance(resp, dict) and resp.get("success", True) and not resp.get("errorMsg")):
            print("order rejected:", resp)
            return False
        return self.ledger.add({**bet, "stake": usd}, mode="live")

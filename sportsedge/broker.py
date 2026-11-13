"""Execution layer. Default = paper. Live is deliberately locked behind several gates."""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
from .ledger import Ledger


class Broker(ABC):
    @abstractmethod
    def place(self, bet: dict) -> bool: ...


class PaperBroker(Broker):
    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def place(self, bet: dict) -> bool:
        return self.ledger.add(bet, mode="paper")


class LiveBroker(Broker):
    """Template only. Implement for an exchange/bookmaker that offers an OFFICIAL API
    (e.g. Betfair Exchange, where permitted in your jurisdiction).
    Do NOT implement by scraping or automating a bookmaker's website: it breaches ToS and risks frozen funds.
    Kenyan retail bookmakers generally have no public betting API; use the slip output and bet manually."""

    def __init__(self, cfg: dict):
        live = cfg.get("live", {})
        if not live.get("enabled"):
            raise RuntimeError("live.enabled is false in config.yaml")
        if os.getenv("LIVE_CONFIRM") != "I_ACCEPT_RISK":
            raise RuntimeError("Set LIVE_CONFIRM=I_ACCEPT_RISK to arm live mode")
        self.manual = live.get("require_manual_approval", True)

    def place(self, bet: dict) -> bool:
        if self.manual and input(f"Place {bet}? type YES: ") != "YES":
            return False
        raise NotImplementedError("Implement the official-API call here, with idempotency + post-trade reconciliation.")

"""SQLite ledger for paper (and later live) bets."""
from __future__ import annotations
import sqlite3, os, datetime as dt


def _now():
    return dt.datetime.utcnow().isoformat()


SCHEMA = """CREATE TABLE IF NOT EXISTS bets(
 id INTEGER PRIMARY KEY, ts TEXT, match_date TEXT, home TEXT, away TEXT,
 market TEXT, selection TEXT, odds REAL, stake REAL, prob REAL, edge REAL,
 status TEXT DEFAULT 'open', pnl REAL, mode TEXT, line REAL, venue TEXT, ref TEXT,
 div TEXT, clv REAL, message_id INTEGER, odds_ts TEXT, book TEXT,
 UNIQUE(match_date, home, away, market, selection, mode))"""


PRED_SCHEMA = """CREATE TABLE IF NOT EXISTS predictions(
 id INTEGER PRIMARY KEY, ts TEXT, match_date TEXT, div TEXT, home TEXT, away TEXT, version TEXT,
 pm_h REAL, pm_d REAL, pm_a REAL, pk_h REAL, pk_d REAL, pk_a REAL, pf_h REAL, pf_d REAL, pf_a REAL,
 odds_h REAL, odds_d REAL, odds_a REAL, outcome TEXT,
 UNIQUE(match_date, div, home, away))"""


class Ledger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(SCHEMA)
        self.db.execute(PRED_SCHEMA)
        for col, typ in (("line", "REAL"), ("venue", "TEXT"), ("ref", "TEXT"),
                         ("div", "TEXT"), ("clv", "REAL"), ("message_id", "INTEGER"),
                         ("odds_ts", "TEXT"), ("book", "TEXT")):   # migrate older ledgers
            try:
                self.db.execute(f"ALTER TABLE bets ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass

    def add(self, bet: dict, mode="paper") -> bool:
        try:
            self.db.execute(
                "INSERT INTO bets(ts,match_date,home,away,market,selection,odds,stake,prob,edge,mode,line,venue,ref,div,odds_ts,book) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (dt.datetime.utcnow().isoformat(), str(bet["date"])[:10], bet["home"], bet["away"], bet.get("market", "1X2"),
                 bet["selection"], bet["odds"], bet["stake"], bet["prob"], bet["edge"], mode,
                 bet.get("line"), bet.get("venue", "bookmaker"), bet.get("ref"), bet.get("div"),
                 bet.get("odds_ts"), bet.get("book")))
            self.db.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def open_bets(self):
        return self.db.execute("SELECT id,match_date,home,away,market,selection,line,odds,stake FROM bets WHERE status='open'").fetchall()

    def find_bet(self, bet: dict, mode: str = "paper"):
        row = self.db.execute(
            "SELECT id FROM bets WHERE match_date=? AND home=? AND away=? AND market=? AND selection=? AND mode=?",
            (str(bet["date"])[:10], bet["home"], bet["away"], bet.get("market", "1X2"),
             bet["selection"], mode)).fetchone()
        return row[0] if row else None

    def settle(self, bid, won: bool, odds, stake):
        pnl = stake * (odds - 1) if won else -stake
        self.db.execute("UPDATE bets SET status=?, pnl=? WHERE id=?", ("won" if won else "lost", pnl, bid))
        self.db.commit()

    def report(self):
        return self.db.execute(
            "SELECT COUNT(*), COALESCE(SUM(stake),0), COALESCE(SUM(pnl),0) FROM bets WHERE status!='open'").fetchone()

    def settled_since(self, days: int):
        import datetime as _dt
        cut = (_dt.datetime.utcnow() - _dt.timedelta(days=days)).date().isoformat()
        return self.db.execute(
            "SELECT match_date,home,away,selection,odds,stake,status,pnl FROM bets "
            "WHERE status!='open' AND match_date>=? ORDER BY match_date", (cut,)).fetchall()

    def log_prediction(self, div, row, pr, version="v0"):
        """Log EVERY pre-match prediction (not just bets). First write wins, so odds/probs are never rewritten after the fact."""
        self.db.execute(
            "INSERT OR IGNORE INTO predictions(ts,match_date,div,home,away,version,pm_h,pm_d,pm_a,pk_h,pk_d,pk_a,pf_h,pf_d,pf_a,odds_h,odds_d,odds_a)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (_now(), str(row["Date"])[:10], div, row["HomeTeam"], row["AwayTeam"], version,
             *map(float, pr["model_raw"]), *map(float, pr["market"]), *map(float, pr["final"]), *map(float, pr["bet_odds"])))
        self.db.commit()

    def settle_predictions(self, hist):
        import pandas as pd
        for pid, d, home, away in self.db.execute(
                "SELECT id,match_date,home,away FROM predictions WHERE outcome IS NULL").fetchall():
            m = hist[(hist.Date == pd.Timestamp(d)) & (hist.HomeTeam == home) & (hist.AwayTeam == away)]
            if len(m):
                self.db.execute("UPDATE predictions SET outcome=? WHERE id=?", (m.iloc[0].FTR, pid))
        self.db.commit()

    def settled_predictions(self):
        import pandas as pd
        return pd.read_sql_query("SELECT * FROM predictions WHERE outcome IS NOT NULL ORDER BY match_date", self.db)

    def settled_bets(self):
        import pandas as pd
        return pd.read_sql_query("SELECT * FROM bets WHERE status!='open' ORDER BY match_date", self.db)

    def settle_pnl(self, bid, pnl: float, clv=None):
        self.db.execute("UPDATE bets SET status=?, pnl=?, clv=? WHERE id=?",
                        ("won" if pnl > 0 else "lost" if pnl < 0 else "push", pnl, clv, bid))
        self.db.commit()

    def set_message_id(self, bid, message_id: int):
        self.db.execute("UPDATE bets SET message_id=? WHERE id=?", (message_id, bid))
        self.db.commit()

    def posted_open_bets(self):
        """Open bets that were published to Telegram (message_id set)."""
        return self.db.execute(
            "SELECT id,match_date,home,away,market,selection,line,odds,message_id FROM bets "
            "WHERE status='open' AND message_id IS NOT NULL").fetchall()

    def unposted_bets(self, since_date: str):
        """Open bets with no message_id yet (dry-run runs, failed posts). id, bet fields, prob, edge, stake."""
        return self.db.execute(
            "SELECT id,match_date,home,away,market,selection,line,odds,stake,prob,edge,venue,div FROM bets "
            "WHERE status='open' AND message_id IS NULL AND match_date>=? ORDER BY match_date", (since_date,)).fetchall()

    def withdraw(self, bid):
        self.db.execute("UPDATE bets SET status='withdrawn', pnl=0 WHERE id=? AND status='open'", (bid,))
        self.db.commit()

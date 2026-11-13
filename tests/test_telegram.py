from sportsedge.telegram import split_message, TelegramPublisher
from sportsedge.report import format_bet, format_digest, format_results


BET = dict(home="A<b>", away="B&Co", date="2026-11-14", div="E0", selection="H", odds=2.2, prob=0.5,
           edge=0.10, stake_pct=0.8, xg=(1.6, 1.1), top_scores=[("1-1", .12), ("2-1", .10), ("1-0", .09)])


def test_split_respects_limit():
    txt = "\n\n".join(["x" * 1500] * 6)
    assert all(len(c) <= 4000 for c in split_message(txt))


def test_html_escaped_and_disclaimer():
    msg = format_digest([BET])
    assert "A&lt;b&gt;" in msg and "B&amp;Co" in msg and "not guaranteed" in msg


def test_no_bets_digest():
    assert "No bets today" in format_digest([])


def test_dry_run_when_no_credentials(monkeypatch, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHANNEL_ID", raising=False)
    assert TelegramPublisher().send("hi") is True
    assert "DRY RUN" in capsys.readouterr().out


def test_results_lists_losses_too():
    rows = [("2026-11-10", "A", "B", "H", 2.0, 10, "won", 10.0), ("2026-11-11", "C", "D", "A", 3.0, 10, "lost", -10.0)]
    msg = format_results(rows, 7, 1000, (2, 20, 0.0))
    assert "✅" in msg and "❌" in msg

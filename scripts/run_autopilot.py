"""Run one autopilot cycle safely from a scheduler (cron / Windows Task Scheduler).

Guards:
- A lock file prevents two overlapping runs (a slow fetch must not race a settle).
- The working directory is forced to the project root, so relative config/data paths resolve.
- A non-zero exit code tells the scheduler the run failed (alerts you, e.g. via Task Scheduler).

Usage:
    python scripts/run_autopilot.py            # daily, from the scheduler
    python scripts/run_autopilot.py --command paper   # run any sportsedge command the same way
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data" / ".autopilot.lock"
STALE_AFTER = 6 * 3600  # a lock older than 6h means a crashed run: take it over


def acquire() -> bool:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        age = time.time() - LOCK.stat().st_mtime
        if age < STALE_AFTER:
            print(f"another autopilot run is active (lock age {age:.0f}s) — skipping")
            return False
        print(f"stale lock found (age {age / 3600:.1f}h, a previous run must have crashed) — taking over")
        LOCK.unlink()
    LOCK.write_text(str(os.getpid()))
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--command", default="autopilot")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if not acquire():
        return 0  # skipped overlap: not an error for the scheduler
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))  # `python scripts/run_autopilot.py` puts scripts/ on sys.path, not the project root
    try:
        from sportsedge.cli import main as cli_main
        import sys as _sys
        args = [a.command]
        if a.force:
            args.append("--force")
        old = _sys.argv
        _sys.argv = ["sportsedge"] + args
        try:
            cli_main()
        finally:
            _sys.argv = old
        return 0
    except SystemExit as e:  # sportsedge uses SystemExit for missing data/config
        print("run failed:", e)
        return 1
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if LOCK.exists():
            LOCK.unlink()


if __name__ == "__main__":
    sys.exit(main())

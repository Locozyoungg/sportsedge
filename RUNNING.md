# Running SportsEdge

Step-by-step guide for installing, running, and scheduling the SportsEdge betting
research agent. All commands below are run from the **project root** (the folder
containing `config.yaml`).

---

## 1. One-time setup

### 1.1 Install Python and dependencies

Requires Python 3.13 (earlier 3.10+ versions usually work). On Windows:

```powershell
cd E:\Projects\sportsedge\extracted\sportsedge

# Create a virtual environment (recommended)
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

(If you get a PowerShell execution-policy error activating the venv, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` first.)

### 1.2 Configure credentials (`.env`)

Copy `.env.example` to `.env` and fill in what you use:

```powershell
Copy-Item .env.example .env
```

| Variable | Needed for | Without it |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Publishing picks to Telegram | Posts are printed as a dry run |
| `TELEGRAM_CHANNEL_ID` | Publishing picks to Telegram | Same — dry run |
| `TELEGRAM_ADMIN_CHAT_ID` | Diagnostics: pauses, health alerts, learning reports | Alerts are skipped |
| `ODDS_API_KEY` | Live multi-book odds before publishing | Uses only the downloaded odds file |
| `LIVE_CONFIRM` | Live real-money orders | Leave **empty** — live trading stays locked |
| `POLYMARKET_*` | Polymarket live orders (untested) | Leave empty; paper comparison still works |

`.env` is git-ignored — never commit it. See the README "Telegram premium
channel" section for how to create the bot and channel.

### 1.3 Sanity check

```powershell
python -m pytest -q        # 42 tests, synthetic data only
python -m sportsedge validate
```

`validate` should print `{'ok': True, 'fatal': [], 'warn': []}`. If it reports
fatal problems, fix the data before running (see Troubleshooting).

---

## 2. Running the bot (daily cycle)

The bot is one command:

```powershell
python scripts/run_autopilot.py
```

This is the scheduler-safe entrypoint (lock-guarded, forces the working
directory, non-zero exit code on failure). It runs one full cycle:

1. **fetch** — download new results + odds + fixtures, then validate them. On
   fatal data problems it writes a `STOP` file and pauses (see §4).
2. **odds** — refresh live multi-book odds (only if `ODDS_API_KEY` is set).
3. **polymarket-scan** — attach Polymarket prices (only if enabled in config).
4. **settle** — settle open paper bets from the new results (with CLV).
5. **post-mortem** — luck-vs-error report (calibration, bias, P&L).
6. **health** — checks the model against the market; writes `STOP` and alerts
   the admin chat if the blend is persistently worse or drawdown too deep.
7. **learn** — challenger search + gated promotion (at most every 7 days).
8. **shadow** — log/settle challenger shadow predictions.
9. **publish** — record new paper bets and post picks to Telegram
   (blocked while a `STOP` file exists).
10. On **Mondays** also: weekly self-evaluation + 7-day results post +
    track-record post.

Run it manually with `python -m sportsedge autopilot` for the same cycle
without the lock guard.

### 2.1 Individual steps (manual / debugging)

```powershell
python -m sportsedge fetch          # data only: results + odds + fixtures
python -m sportsedge settle         # settle open paper bets
python -m sportsedge predict        # today's slip, printed only (records nothing)
python -m sportsedge paper          # record paper bets (no Telegram post)
python -m sportsedge publish        # record + post picks to Telegram
python -m sportsedge withdraw       # re-check posted picks vs fresh odds; edit/void if edge is gone
python -m sportsedge results --days 7   # post settled results to the channel
python -m sportsedge monthly --days 30   # subscriber report
python -m sportsedge postmortem     # luck-vs-error report
python -m sportsedge health         # health check (writes STOP if unhealthy)
python -m sportsedge learn --force  # challenger search + gated promotion
python -m sportsedge shadow         # challenger shadow research
python -m sportsedge telegram-test  # verify Telegram wiring
```

### 2.2 Dashboard

```powershell
streamlit run sportsedge/dashboard.py
```

Read-only view: bankroll curve, drawdown, CLV, calibration, per-league ROI,
open bets, learning history. Works as soon as the first paper bets settle.

---

## 3. Scheduling (set once, then it runs itself)

The bot is designed for **one autopilot run per day**.

### Windows Task Scheduler

1. Open **Task Scheduler** → **Create Basic Task**.
2. Name: `SportsEdge autopilot`. Trigger: **Daily**, pick a time after matches
   finish (e.g. 23:30 — the data source usually updates by then).
3. Action: **Start a program**:
   - Program/script: `C:\Path\To\Python\python.exe`
     (find it with `(Get-Command python).Source` from an activated venv — use
     the venv python, e.g. `.venv\Scripts\python.exe`)
   - Arguments: `scripts/run_autopilot.py`
   - Start in: `E:\Projects\sportsedge\extracted\sportsedge`  ← must be set
4. On the **Settings** tab, tick *"Run task as soon as possible after a
   scheduled start is missed"*.
5. A **non-zero exit code = failed run** — add a failure action (e.g. log to a
   file or alert) so silent failures don't go unnoticed.

### Linux/macOS cron

```cron
30 23 * * * cd /path/to/sportsedge && /path/to/.venv/bin/python scripts/run_autopilot.py >> logs/autopilot.log 2>&1
```

### Docker

```bash
docker build -t sportsedge .
docker run --rm -v "$PWD/data":/app/data -v "$PWD/models":/app/models --env-file .env sportsedge
```

---

## 4. The STOP file (paused pipeline)

If a file named `STOP` exists in the project root, the bot **will not record
or publish** anything (data collection and settling still run). It is created
by:

- **`health`** — model significantly worse than the market, or drawdown too
  deep. An alert is sent to `TELEGRAM_ADMIN_CHAT_ID`.
- **data validation failure** in `fetch`/`autopilot`.
- **you, manually** — the file itself is the pause switch.

To resume: read the file, fix the cause, then delete it.

```powershell
Remove-Item STOP
```

If a validation failure caused it, run `python -m sportsedge validate` again
and inspect the fatal items first — deleting the file without fixing the data
just means the next autopilot pauses again.

---

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `another autopilot run is active — skipping` | A run is in progress or a previous run crashed < 6 h ago. Locks expire after 6 h and are taken over automatically. |
| `no upcoming fixtures` | The fixtures file is empty/outdated. Run `python -m sportsedge fetch` (or `python -m sportsedge odds` with `ODDS_API_KEY`). |
| Posts print instead of sending to Telegram | `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHANNEL_ID` unset, or bot not admin of the channel. Test with `python -m sportsedge telegram-test`. |
| `STOP file present: autopilot paused` | See §4. |
| Live orders never fire | Intended. `live.enabled: false` in config, `LIVE_CONFIRM` empty, geoblock, and manual approval all lock live trading. |
| `run failed: SystemExit` from the scheduler script | A step failed (missing data/config). Check the log output. |
| Autopilot is slow (several minutes) | Normal — fetch + model refit + learning grid. Let it finish; the lock prevents overlap. |

---

## 6. Safety reminders

- This is a **paper trading** bot. The honest backtest verdict (see
  `BACKTEST_REPORT.md`) is that the model does not beat the market — treat
  published picks as research, never as financial advice.
- Live execution is deliberately locked (config + `LIVE_CONFIRM` env +
  manual approval). Do not unlock it without 300+ settled paper bets with
  positive CLV.
- Never delete `STOP` without reading it — it is the bot's kill switch.
- Only bet money you can afford to lose. Betting can be addictive: set deposit
  limits.

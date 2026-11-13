# SportsEdge paper-trading pipeline (autopilot). Live trading stays off; see README.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt requirements-live.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-live.txt

COPY . .

# data/, models/, reports/ are written at runtime; keep them out of the image layers.
VOLUME ["/app/data", "/app/models", "/app/reports"]

# Scheduler entrypoint: one autopilot cycle (fetch -> settle -> health -> learn -> shadow -> publish).
# Run on a cron/task schedule with the .env mounted for ODDS_API_KEY / TELEGRAM_*.
CMD ["python", "scripts/run_autopilot.py"]

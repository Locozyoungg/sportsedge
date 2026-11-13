"""Telegram publisher. Dry-run (prints) unless TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID are set."""
from __future__ import annotations
import os, time
import requests

API = "https://api.telegram.org/bot{token}/{method}"
LIMIT = 4000  # Telegram hard limit is 4096


def split_message(text: str, limit: int = LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks, cur = [], ""
    for block in text.split("\n\n"):
        if len(cur) + len(block) + 2 > limit and cur:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n\n{block}" if cur else block
    if cur:
        chunks.append(cur)
    return chunks


class TelegramPublisher:
    def __init__(self, token: str | None = None, chat_id: str | None = None, dry_run: bool | None = None,
                 chat_env: str = "TELEGRAM_CHANNEL_ID"):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv(chat_env)
        self.dry_run = (not (self.token and self.chat_id)) if dry_run is None else dry_run

    def _call(self, method: str, payload: dict, retries: int = 3):
        """Returns the response dict on success, None on failure. Honours 429 retry_after."""
        for attempt in range(retries):
            try:
                r = requests.post(API.format(token=self.token, method=method), json=payload, timeout=20)
                if r.status_code == 429:  # rate limited: honour Telegram's retry_after
                    time.sleep(r.json().get("parameters", {}).get("retry_after", 3) + 1)
                    continue
                if r.ok:
                    return r.json()
                print("telegram error:", r.status_code, r.text[:200])
                return None
            except requests.RequestException as e:
                print("telegram network error:", e)
                time.sleep(2 ** attempt)
        return None

    def post(self, text: str) -> list[int]:
        """Send each chunk; returns the list of message ids ([] if nothing could be sent)."""
        ids = []
        for chunk in split_message(text):
            if self.dry_run:
                print("\n----- [DRY RUN: not sent to Telegram] -----\n" + chunk + "\n-------------------------------------------")
                ids.append(-1)  # placeholder so dry-run pipelines can still store "posted" markers
                continue
            r = self._call("sendMessage", {"chat_id": self.chat_id, "text": chunk, "parse_mode": "HTML",
                                           "disable_web_page_preview": True})
            if r and r.get("message_id"):
                ids.append(int(r["message_id"]))
                time.sleep(1)  # stay well under channel rate limits
            else:
                return []  # partial failure: caller must not mark anything as posted
        return ids

    def edit(self, message_id: int, text: str) -> bool:
        """Edit a previously sent message (e.g. 'odds moved, pick withdrawn')."""
        if self.dry_run:
            print("\n----- [DRY RUN: not sent to Telegram] -----\nedit msg " + str(message_id) + ":\n" + text[:400])
            return True
        r = self._call("editMessageText", {"chat_id": self.chat_id, "message_id": message_id,
                                           "text": text[:4096], "parse_mode": "HTML",
                                           "disable_web_page_preview": True})
        return bool(r)

    def send(self, text: str) -> bool:
        return len(self.post(text)) > 0

"""Telegram interface for generating and delivering reports on demand.

Run separately from main.py: python telegram_reports.py
Requires: python -m pip install requests
"""
from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
import re
import time

import settings

LOGGER = logging.getLogger("reports_nas.telegram")
API = "https://api.telegram.org/bot"
COMMAND_RE = re.compile(r"^/([a-zA-Z0-9_]+)(?:@[a-zA-Z0-9_]+)?(?:\s+(.*))?$")
MAX_DOCUMENT_BYTES = 49 * 1024 * 1024  # Conservative Bot API upload limit.


def bot_token():
    return os.environ.get("REPORTS_TELEGRAM_TOKEN", "").strip() or str(
        settings.TELEGRAM_BOT_TOKEN
    ).strip()


def request_api(method, *, data=None, files=None, timeout=60):
    import requests

    token = bot_token()
    if not token:
        raise RuntimeError("Set REPORTS_TELEGRAM_TOKEN or TELEGRAM_BOT_TOKEN.")
    response = requests.post(
        f"{API}{token}/{method}", data=data, files=files, timeout=timeout
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram {method} failed: {payload.get('description')}")
    return payload["result"]


def send_message(text):
    return request_api(
        "sendMessage", data={"chat_id": str(settings.TELEGRAM_CHAT_ID), "text": text}
    )


def send_document(path):
    path = Path(path)
    if path.stat().st_size > MAX_DOCUMENT_BYTES:
        raise ValueError(
            f"{path.name} exceeds the 49 MiB direct upload limit. "
            "Reduce the file size or distribute it through another storage service."
        )
    with path.open("rb") as handle:
        return request_api(
            "sendDocument",
            data={"chat_id": str(settings.TELEGRAM_CHAT_ID)},
            files={"document": (path.name, handle)},
            timeout=180,
        )


def help_text():
    lines = [
        "Rat Village Reports",
        "",
        "/help - Show commands and available reports",
        "/list - List configured report names",
        "/report NAME - Generate and send one report from prepared data",
        "/report NAME fresh - Refresh source data first, then generate",
        "/refresh - Rebuild prepared_trial_data.csv from NAS source files",
        "/ping - Check whether the report bot is online",
        "",
        "Shortcuts: /NAME or just NAME also request a report.",
        "",
        "Available reports:",
    ]
    for key, item in settings.TELEGRAM_REPORTS.items():
        lines.append(f"/{key} - {item.get('description', key)}")
    lines.append("")
    lines.append("Example: /report weekly fresh")
    return "\n".join(lines)


def prepared_path():
    return Path(settings.OUTPUT_DIRECTORY) / settings.OUTPUT_FILENAME


def load_data(*, fresh=False):
    if not fresh and bool(settings.TELEGRAM_USE_PREPARED_CACHE) and prepared_path().is_file():
        from report_utils import read_sorted_data

        LOGGER.info("Reading prepared data from %s", prepared_path())
        return read_sorted_data()

    from prepare_data import prepare_data

    LOGGER.info("Preparing fresh data from NAS sources")
    data = prepare_data(apply_filters=False)
    target = prepared_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(target, sep=settings.CSV_SEPARATOR, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved prepared data: %s (%s rows)", target, len(data))
    return data


def generate_report(name, *, fresh=False):
    """Generate only the configured report and return its expected files.

    Only trusted, hard-coded configuration entries can select a module. Telegram
    input is never treated as a Python module or filesystem path.
    """
    entry = settings.TELEGRAM_REPORTS[name]
    module_name = entry["module"]
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", module_name):
        raise ValueError("Invalid report module configured in settings.py")

    # Resolve and verify configured outputs before importing any module.
    base = Path(settings.REPORT_DIRECTORY).resolve()
    outputs = []
    for filename in entry["files"]:
        path = (base / filename).resolve()
        if not path.is_relative_to(base) or path == base:
            raise ValueError("Report output path must stay in REPORT_DIRECTORY.")
        outputs.append(path)

    data = load_data(fresh=fresh)
    module = importlib.import_module(module_name)
    started = time.time()
    result = module.generate(data)
    if result is None:
        raise RuntimeError(f"Report '{name}' is disabled in settings.py.")
    # Avoid sending an old report when a generator silently failed to rewrite it.
    for path in outputs:
        if not path.is_file() or path.stat().st_mtime < started - 2:
            raise FileNotFoundError(f"Expected newly generated report not found: {path}")
    return outputs


def handle_message(message):
    chat = message.get("chat", {})
    if str(chat.get("id", "")) != str(settings.TELEGRAM_CHAT_ID):
        LOGGER.warning("Rejected command from unauthorized Telegram chat %s", chat.get("id"))
        return
    raw = str(message.get("text", "")).strip()
    if not raw:
        return

    command = COMMAND_RE.match(raw)
    if raw.startswith("/") and not command:
        send_message("Invalid command. Send /help for usage.")
        return
    if command:
        name = command.group(1).lower()
        arguments = (command.group(2) or "").strip().lower().split()
    else:
        words = raw.lower().split()
        name, arguments = words[0], words[1:]

    if name in ("help", "start", "list"):
        send_message(help_text())
        return
    if name == "ping":
        send_message("Reports bot is online.")
        return
    if name == "refresh":
        send_message("Refreshing prepared data from NAS files...")
        data = load_data(fresh=True)
        send_message(f"Prepared data updated: {len(data):,} trial rows.")
        return

    if name == "report":
        if not arguments:
            send_message(help_text())
            return
        name, arguments = arguments[0], arguments[1:]

    if name not in settings.TELEGRAM_REPORTS:
        send_message(f"Unknown report '{name}'. Send /help to see available reports.")
        return
    if arguments and arguments != ["fresh"]:
        send_message(f"Usage: /report {name} [fresh]")
        return
    fresh = arguments == ["fresh"]
    send_message(f"Generating {name} report" + (" with fresh data..." if fresh else "..."))
    paths = generate_report(name, fresh=fresh)
    for path in paths:
        send_document(path)
    send_message(f"Report '{name}' completed. Sent {len(paths)} file(s).")


def poll_forever():
    """Long-poll Telegram with bounded retries and no background report threads."""
    offset = None
    while True:
        try:
            params = {
                "timeout": int(settings.TELEGRAM_POLL_TIMEOUT),
                "allowed_updates": '["message"]',
            }
            if offset is not None:
                params["offset"] = offset
            updates = request_api(
                "getUpdates",
                data=params,
                timeout=int(settings.TELEGRAM_POLL_TIMEOUT) + 15,
            )
            for update in updates:
                offset = int(update["update_id"]) + 1
                try:
                    handle_message(update.get("message", {}))
                except Exception:
                    LOGGER.exception("Report request failed")
                    try:
                        send_message("Report request failed. Check the reports bot log.")
                    except Exception:
                        LOGGER.exception("Could not send failure notification")
        except KeyboardInterrupt:
            raise
        except Exception:
            LOGGER.exception("Telegram polling failed; retrying")
            time.sleep(10)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not settings.TELEGRAM_ENABLED:
        raise SystemExit("Set TELEGRAM_ENABLED = True in settings.py.")
    if not bot_token() or not str(settings.TELEGRAM_CHAT_ID).strip():
        raise SystemExit("Configure the Telegram bot token and authorized chat ID.")
    if not settings.TELEGRAM_REPORTS:
        raise SystemExit("TELEGRAM_REPORTS is empty.")
    LOGGER.info("Telegram reports bot started")
    poll_forever()


if __name__ == "__main__":
    main()

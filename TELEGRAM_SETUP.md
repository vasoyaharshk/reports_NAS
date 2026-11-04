# Telegram reports bot

The bot is separate from `main.py`: it waits for a message, generates only
the selected report, then sends the configured PDF/CSV/XLSX files to Telegram.
The normal bulk-report command remains unchanged.

## Setup (Windows)

1. Install `python -m pip install requests pandas openpyxl matplotlib` in the
   environment you already use for `reports_NAS`.
2. Edit `settings.py`:
   - Set `TELEGRAM_ENABLED = True`.
   - Set `TELEGRAM_CHAT_ID` to the exact chat ID that may request reports.
   - Set `TELEGRAM_BOT_TOKEN` locally, or preferably set the
     `REPORTS_TELEGRAM_TOKEN` environment variable on your computer.
   - Add or rename entries in `TELEGRAM_REPORTS`. The entry key is the
     command name; `module` is an existing project module with
     `generate(data)`; `files` contains paths relative to
     `REPORT_DIRECTORY`; `description` appears in /help.
3. Run `python telegram_reports.py` and open a private chat with your bot.
4. Send `/ping`, `/help`, then `/weekly` or `/report weekly`.

**Never commit the real bot token.** `settings.py` is tracked in this public
repository: use the environment variable for a real token. For a per-user
Windows setting, use `setx REPORTS_TELEGRAM_TOKEN "YOUR_BOT_TOKEN"`, then open
a **new** terminal. Do not share screenshots showing your token.

Only the configured chat ID can request reports or receive documents. For a
personal bot, configure your individual private-chat ID (not a public group).
To obtain your chat ID, send a message to your bot and use Telegram's `getUpdates`
API, or a reputable ID lookup bot. Do not expose the bot token or raw API URL in
screenshots, logs, or chats.

## Available commands

- `/help` or `/list`: list names, descriptions, and instructions.
- `/ping`: verify that the bot process responds.
- `/weekly`, `weekly`, or `/report weekly`: generate the weekly report.
- `/report status`, `/report trials`, `/report period`, etc.: request any
  configured name.
- `/report weekly fresh`: rebuild preparation from NAS source files, then
  generate the selected report.
- `/refresh`: update `output/prepared_trial_data.csv` without making a PDF.

By default, reports reuse `prepared_trial_data.csv` when it exists, to avoid
repeating expensive preparation for every Telegram request. If the file does not
exist, preparation runs automatically. To always prepare fresh data, set
`TELEGRAM_USE_PREPARED_CACHE = False`. The report-generating call still executes
on **every** request even when preparation is cached.

The bot sends only files listed in the selected report entry. New report
generators can be added to the dictionary after they expose `generate(data)`
and write the configured filenames. Direct Telegram file uploads are limited to
approximately 50 MB; the bot enforces a conservative 49 MiB limit. Large
`prepared_trial_data.csv` files should **not** be sent through Telegram.

## Keep the bot running automatically on Windows

**Recommended: Windows Task Scheduler** on a Windows PC that stays powered on,
connected to the NAS and Internet. Use **Create Task** rather than Basic Task.

- **General:** choose a dedicated Windows account that can read `H:\\NAS`
  and use `Run only when user is logged on` if the NAS depends on the mapped
  `H:` drive. This is the most predictable choice for a mapped drive.
- **Triggers:** `At log on` of that account; optionally delay by 30 to 60
  seconds to allow the NAS share to mount.
- **Action:** `Start a program`; program is the absolute path to
  `python.exe` in your Conda environment; argument is
  `telegram_reports.py`; **Start in** is `H:\\reports_NAS` (no quotes).
- **Settings:** if the task fails, restart every 1 minute (try indefinitely),
  prevent overlapping instances, and disable the option that stops a task
  after 3 days.
- **Conditions:** disable idle-only triggers and settings that would stop the
  task when the machine leaves idle; prevent sleep when you need 24/7 access.

Verify the mapped `H:` drive in the same account and session as the task.
If you want the bot to run *without any Windows login*, use a UNC NAS path
(`\\\\server\\share`) and stored service-account permissions, not a
per-user mapped `H:` drive; then configure the task to run whether or not
the user is logged on. A machine that is switched off, sleeping, or unable to
access Telegram/NAS cannot answer requests. A small always-on NAS/docker host
or a server is another option, but paths in `settings.py` must be adapted
for that machine.

**Important:** only run one polling instance of a bot token at a time.
If Academy uses the same Telegram token, create a *separate bot token* for
reports or the two long-polling clients will conflict.

## Deployment

After merging the feature branch, run `git pull` on the machine hosting
`reports_NAS`. The bot is not remotely started by GitHub. You must launch
`python telegram_reports.py` on your PC and configure Task Scheduler
separately. Monitor the Windows task history and application logs.

## Synology NAS deployment (preferred)

Use Container Manager on your Synology NAS, rather than Windows Task Scheduler.
The repository includes `Dockerfile.telegram` and `docker-compose.telegram.yml`.
Set the NAS share mount on the left side of the volumes to the actual location
of your files. `/volume1/NAS` is only an example. The container sees `/data`.
Create persistent directories for reports and prepared output before starting.

Create a local `.env` file next to the Compose file containing the two keys
`REPORTS_TELEGRAM_TOKEN` and `REPORTS_TELEGRAM_CHAT_ID` with your actual values.
Keep this file private and never commit it to Git. Import the Compose project
in Synology Container Manager, or run `docker compose -f docker-compose.telegram.yml up -d --build`
from the repository folder. The service uses `restart: unless-stopped`, so it
restarts when Docker starts after a NAS reboot. The NAS must have Internet
access, and the container must be able to read the data share and write the
prepared-data and reports directories. Some data preparation steps also write
merged environment files into the data share.

The Docker environment variables override Windows-only `H:` paths in settings.
You do not need the Windows PC to remain on. Check Container Manager logs
for errors, and send `/ping` followed by `/report weekly` to validate.

If your Synology volume is not `volume1`, update the paths accordingly.

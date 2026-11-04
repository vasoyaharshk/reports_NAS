"""Configuration for reports_NAS.

Edit the paths below on the NAS host before running prepare_data.py.
CSV inputs use the semicolon separator used by the source exports.
"""
import os
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parent
# The NAS data folders are outside the cloned repository.
DATA_DIRECTORY = Path(os.environ.get('REPORTS_DATA_DIRECTORY', r"H:\NAS"))

# Rat metadata workbook.
RAT_HEALTH_FILE = DATA_DIRECTORY / "rat_health_sheet" / "rat_health_LE.xlsx"
RAT_DETAILS_SHEET = "rat_details"
RAT_WEIGHT_SHEETS = {
    1: "weights_females_rv1",
    2: "weights_males_rv2",
    3: "weights_females_rv3",
}
DAILY_FOOD_WATER_SHEET = "daily_food_water"
FOOD_WATER_SHEETS = {
    1: "food_water_rv1",
    2: "food_water_rv2",
    3: "food_water_rv3",
}
# Two-week periods are aligned to Mondays from this anchor date.
BIWEEKLY_PERIOD_ANCHOR = "2024-01-01"

# Each path points to a folder. Files are discovered by filename prefix:
# early_training*.csv, global*.csv, and global_trials_wm*.csv.
VILLAGE_INPUTS = {
    1: {
        "early_training": DATA_DIRECTORY / "global_trials_rv1",
        "global_trials_pi": DATA_DIRECTORY / "global_trials_rv1",
        "working_memory": DATA_DIRECTORY / "global_trials_rv1",
    },
    2: {
        "early_training": DATA_DIRECTORY / "global_trials_rv2",
        "global_trials_pi": DATA_DIRECTORY / "global_trials_rv2",
        "working_memory": None,
    },
    3: {
        "early_training": DATA_DIRECTORY / "global_trials_rv3",
        "global_trials_pi": DATA_DIRECTORY / "global_trials_rv3",
        "working_memory": None,
    },
}

# Task switches: True includes that task family in the sorted output.
Working_Memory = True
Probability = True
Cognitive_bias = True

# Rows whose task name does not match one of the task families above are retained.
# This keeps new or renamed tasks visible until their category is added explicitly.
INCLUDE_UNCLASSIFIED_TASKS = True

# Kept configurable to mirror the exclusions in the current R sorting script.
EXCLUDED_SUBJECTS = ("m1", "none", "None", "m2", "m3")

# Optional workbook containing sheets named ratvillage1, ratvillage2, ratvillage3
# with date and inactive_hours columns. Leave as None if it is not available.
INACTIVE_HOURS_FILE = None

OUTPUT_DIRECTORY = Path(os.environ.get("REPORTS_OUTPUT_DIRECTORY", str(PROJECT_DIRECTORY / "output")))
OUTPUT_FILENAME = "prepared_trial_data.csv"
CSV_SEPARATOR = ";"

# PDF and daily trials-per-task output location.
REPORT_DIRECTORY = Path(os.environ.get("REPORTS_REPORT_DIRECTORY", str(PROJECT_DIRECTORY / "reports")))
TRIALS_REPORT_DAYS = 30
# Number of most recent active trial dates to include per village in the
# combined status/activity report.
STATUS_REPORT_DAYS = 30
REPORT_TIMEZONE = "Europe/Madrid"
# Optional note shown on report title pages, such as "Sampled rows only".
REPORT_DATA_NOTE = ""

# RV2/RV3 before-and-after comparison, with boundaries in Europe/Madrid time.
# Period starts are inclusive and ends are exclusive. The final None end means
# through the latest trial in the sorted dataset.
PERIOD_COMPARISON_TIMEZONE = "Europe/Madrid"
PERIOD_COMPARISON_SESSION_COLUMN = None  # Auto-detect; set explicitly if needed.
PERIOD_COMPARISON_PERIODS = [
    {"period": "Before", "start": "2026-10-04 00:00:00", "end": "2026-10-19 00:00:00"},
    {"period": "20ul 5%", "start": "2026-10-19 12:00:00", "end": "2026-10-22 16:00:00"},
    {"period": "20ul 10%", "start": "2026-10-22 16:00:00", "end": "2026-10-26 13:00:00"},
    {"period": "75ul 10%", "start": "2026-10-26 13:00:00", "end": None},
]
PERIOD_COMPARISON_OUTPUT_FILENAME = "RV2_RV3_Trials_Sessions_By_Period.xlsx"

# Each village points to its events folder. Every events*.csv file is loaded.
EVENT_DIRECTORIES = {
    1: DATA_DIRECTORY / "events_rv1",
    2: DATA_DIRECTORY / "events_rv2",
    3: DATA_DIRECTORY / "events_rv3",
}

# Environment source folders are read only. Merged, de-duplicated files are
# saved separately in H:\\NAS\\environment.
ENVIRONMENT_DIRECTORIES = {
    1: None,
    2: DATA_DIRECTORY / "environment_rv2",
    3: DATA_DIRECTORY / "environment_rv3",
}
ENVIRONMENT_MERGED_DIRECTORY = DATA_DIRECTORY / "environment"
ENVIRONMENT_EXCEL_SHEET = "Environment"
ENVIRONMENT_TIMEZONE = "Europe/Madrid"

# Academy event timestamps use a fixed UTC+1 offset year round.
EVENT_TIMEZONE = "Etc/GMT-1"
EVENT_SUBJECT_ALIASES = {"chand": "chandler"}
EVENT_NON_ANIMAL_SUBJECTS = (
    "academy", "none", "nan", "manual", "manual1", "manual2",
)

# Telegram report request bot.
# Set both fields locally before starting telegram_reports.py.
# Prefer the REPORTS_TELEGRAM_TOKEN environment variable for the secret.
TELEGRAM_BOT_TOKEN = ""  # Or set REPORTS_TELEGRAM_TOKEN on the host.
TELEGRAM_CHAT_ID = ""  # Only this chat is allowed to request and receive reports.
TELEGRAM_ENABLED = False
TELEGRAM_POLL_TIMEOUT = 25

# The keys below are the report names sent to the bot. Add, remove, or
# rename keys freely. Each module must expose generate(data).
# "files" controls which generated files are sent, relative to REPORT_DIRECTORY.
# A command can also be sent as /report weekly or simply /weekly.
TELEGRAM_REPORTS = {
    "weekly": {
        "description": "Weekly trials, occupancy and comparisons across villages",
        "module": "Weekly_Daily_Trials",
        "files": ["Weekly_Daily_Trials.pdf", "Weekly_Daily_Trials.csv"],
    },
    "status": {
        "description": "Rat Village status and activity",
        "module": "Status_Activity_Report",
        "files": ["Status_and_Activity_Report.pdf"],
    },
    "early": {
        "description": "Early Training report",
        "module": "Early_Training",
        "files": ["Early_Training_Report.pdf"],
    },
    "memory": {
        "description": "Working Memory report",
        "module": "Working_Memory_Report",
        "files": ["Working_Memory_Report.pdf"],
    },
    "bias": {
        "description": "Cognitive Bias report",
        "module": "Cognitive_bias",
        "files": ["Cognitive_Bias_Report.pdf"],
    },
    "probability": {
        "description": "Probability report",
        "module": "Probability",
        "files": ["Probability_Report.pdf"],
    },
    "trials": {
        "description": "Trials per Task PDF and daily CSV",
        "module": "Trials_Per_Task",
        "files": ["Trials_Per_Task_Report.pdf", "Trials_Per_Task_Daily.csv"],
    },
    "period": {
        "description": "RV2 and RV3 period comparison workbook",
        "module": "Period_Comparison",
        "files": [PERIOD_COMPARISON_OUTPUT_FILENAME],
    },
}

# Use a cached prepared_trial_data.csv when available. /refresh runs
# prepare_data(apply_filters=False) and saves the updated CSV before the report.
# /report NAME fresh forces preparation for that single request.
TELEGRAM_USE_PREPARED_CACHE = True

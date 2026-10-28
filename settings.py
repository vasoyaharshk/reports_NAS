"""Configuration for reports_NAS.

Edit the paths below on the NAS host before running prepare_data.py.
CSV inputs use the semicolon separator used by the source exports.
"""
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_DIRECTORY / "Data"

# Data files are grouped by village in the Git repository's Data directory.
# RV1 currently has WM global data only; early-training and PI global files are absent.
VILLAGE_INPUTS = {
    1: {
        "early_training": None,
        "global_trials_pi": None,
        "working_memory": DATA_DIRECTORY / "ratvilalge01" / "global_trials" / "global_trials_wm.csv",
    },
    2: {
        "early_training": DATA_DIRECTORY / "ratvilalge02" / "global_trials" / "early_training_rv2.csv",
        "global_trials_pi": DATA_DIRECTORY / "ratvilalge02" / "global_trials" / "global_trials_pi_rv2.csv",
        "working_memory": None,
    },
    3: {
        "early_training": DATA_DIRECTORY / "ratvillage03" / "global_trials" / "early_training_rv3.csv",
        "global_trials_pi": DATA_DIRECTORY / "ratvillage03" / "global_trials" / "global_trials_pi_rv3.csv",
        "working_memory": None,
    },
}

# Task switches: True includes that task family in the sorted output.
Working_Memory = False
Probability = True
Cognitive_bias = True

# Rows whose task name does not match one of the task families above are retained.
# This keeps new or renamed tasks visible until their category is added explicitly.
INCLUDE_UNCLASSIFIED_TASKS = True

# Kept configurable to mirror the exclusions in the current R sorting script.
EXCLUDED_SUBJECTS = ("m2", "m3", "luna", "rachel", "geralt", "erza", "ross")

# Optional workbook containing sheets named ratvillage1, ratvillage2, ratvillage3
# with date and inactive_hours columns. Leave as None if it is not available.
INACTIVE_HOURS_FILE = None

OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "output"
OUTPUT_FILENAME = "prepared_trial_data.csv"
CSV_SEPARATOR = ";"

# PDF and daily trials-per-task output location.
REPORT_DIRECTORY = PROJECT_DIRECTORY / "reports"
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

# Explicit village mapping is used because event filenames may be renamed.
# Each value accepts one path or a list of paths from the same village.
EVENT_FILES = {
    1: [],
    2: [
        DATA_DIRECTORY / "ratvilalge02" / "events" / "events.csv",
        DATA_DIRECTORY / "ratvilalge02" / "events" / "events_20251001_rv2.csv",
        DATA_DIRECTORY / "ratvilalge02" / "events" / "events_20260402_rv2.csv",
    ],
    3: [DATA_DIRECTORY / "ratvillage03" / "events" / "events.csv"],
}

# Separate input folders keep RV2's multi-sensor master exports and RV3's
# middle-sensor exports independent. The merge command reads CSV and Excel files.
ENVIRONMENT_DIRECTORIES = {
    1: None,
    2: DATA_DIRECTORY / "ratvilalge02" / "environment",
    3: DATA_DIRECTORY / "ratvillage03" / "environment",
}
ENVIRONMENT_EXCEL_SHEET = "Environment"
ENVIRONMENT_TIMEZONE = "Europe/Madrid"
DELETE_ENVIRONMENT_INPUTS_AFTER_MERGE = True

# Academy event dates are written by datetime.now() as local wall time
# (YYYY/MM/DD HH:MM:SS), without a UTC offset.
EVENT_TIMEZONE = "Europe/Madrid"
EVENT_SUBJECT_ALIASES = {"chand": "chandler"}
EVENT_NON_ANIMAL_SUBJECTS = (
    "academy", "none", "nan", "manual", "manual1", "manual2",
)

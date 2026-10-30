"""Configuration for reports_NAS.

Edit the paths below on the NAS host before running prepare_data.py.
CSV inputs use the semicolon separator used by the source exports.
"""
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parent
# The NAS data folders are outside the cloned repository.
DATA_DIRECTORY = Path(r"H:\NAS")

# Rat metadata workbook.
RAT_HEALTH_FILE = DATA_DIRECTORY / "rat_health_sheet" / "rat_health_LE.xlsx"
RAT_DETAILS_SHEET = "rat_details"

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
EXCLUDED_SUBJECTS = ("m2", "m3", "luna", "rachel", "geralt", "erza", "ross", "chand")

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

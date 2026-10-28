"""Configuration for reports_NAS.

Edit the paths below on the NAS host before running data_sorting.py.
CSV inputs use the semicolon separator used by the source exports.
"""
from pathlib import Path

# Data files are configured separately because each Rat Village has its own NAS path.
VILLAGE_INPUTS = {
    1: {
        "early_training": Path("/path/to/ratvillage1/early_training_rv1.csv"),
        "global_trials_pi": Path("/path/to/ratvillage1/global_trials_pi_rv1.csv"),
        # Working memory was only run in RV1; it is loaded only when enabled.
        "working_memory": Path("/path/to/ratvillage1/global_trials_wm.csv"),
    },
    2: {
        "early_training": Path("/path/to/ratvillage2/early_training_rv2.csv"),
        "global_trials_pi": Path("/path/to/ratvillage2/global_trials_pi_rv2.csv"),
        "working_memory": None,
    },
    3: {
        "early_training": Path("/path/to/ratvillage3/early_training_rv3.csv"),
        "global_trials_pi": Path("/path/to/ratvillage3/global_trials_pi_rv3.csv"),
        "working_memory": None,
    },
}

# Task switches: True generates that task-family report.
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

OUTPUT_DIRECTORY = Path("/path/to/reports_NAS/output")
OUTPUT_FILENAME = "sorted_trials.csv"
CSV_SEPARATOR = ";"

# PDF and daily trials-per-task output location.
REPORT_DIRECTORY = Path("/path/to/reports_NAS/reports")
TRIALS_REPORT_DAYS = 30

# RV2/RV3 before-and-after comparison, with boundaries in Europe/Madrid time.
# Period starts are inclusive and ends are exclusive. The final None end means
# through the latest trial in the sorted dataset.
PERIOD_COMPARISON_TIMEZONE = "Europe/Madrid"
PERIOD_COMPARISON_SESSION_COLUMN = None  # Auto-detect; set explicitly if needed.
PERIOD_COMPARISON_PERIODS = [
    {"period": "Before", "start": "2026-10-04 00:00:00", "end": "2026-10-19 00:00:00"},
    {"period": "20ul 5%", "start": "2026-10-19 12:00:00", "end": "2026-10-22 16:00:00"},
    {"period": "20ul 10%", "start": "2026-10-22 16:00:00", "end": None},
]
PERIOD_COMPARISON_OUTPUT_FILENAME = "RV2_RV3_Trials_Sessions_By_Period.xlsx"

# Explicit village mapping is used because event filenames may be renamed.
# Each value accepts one path or a list of paths from the same village.
EVENT_FILES = {
    1: [],  # Example: [Path("/nas/RV1/events.csv"), Path("/nas/RV1/events_20261020.csv")]
    2: [],
    3: [],
}

ENVIRONMENT_FILES = {
    1: [],
    2: [],  # Example: [Path("/nas/RV2/environment_consolidated.xlsx")]
    3: [],
}
ENVIRONMENT_SHEET = "Environment"
ENVIRONMENT_TIMEZONE = "Europe/Madrid"  # Excel timestamps have no timezone metadata.
ENVIRONMENT_MAX_GAP_SECONDS = 15

# Academy event dates are written by datetime.now() as local wall time
# (YYYY/MM/DD HH:MM:SS), without a UTC offset.
EVENT_TIMEZONE = "Europe/Madrid"
EVENT_SUBJECT_ALIASES = {"chand": "chandler"}
EVENT_NON_ANIMAL_SUBJECTS = (
    "academy", "none", "nan", "manual", "manual1", "manual2",
)

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

"""Generate the separate report for early-training source files."""
from pathlib import Path
import settings
from report_utils import write_task_report


def generate(data):
    if "dataset" in data:
        early = data.loc[data["dataset"].astype("string").str.startswith("early_training", na=False)].copy()
    else:
        early = data.iloc[0:0].copy()
    path = Path(settings.REPORT_DIRECTORY) / "Early_Training_Report.pdf"
    return write_task_report(early, "early_training", "Early Training", path)

"""Generate the Probability task report."""
from pathlib import Path
import settings
from report_utils import enabled_task_data, write_task_report


def generate(data):
    selected = enabled_task_data(data, "probability", "Probability")
    if selected is None:
        return None
    path = Path(settings.REPORT_DIRECTORY) / "Probability_Report.pdf"
    return write_task_report(selected, "probability", "Probability", path)

"""Generate the Working Memory report (RV1 task family)."""
from pathlib import Path
import settings
from report_utils import enabled_task_data, write_task_report


def generate(data):
    selected = enabled_task_data(data, "working_memory", "Working_Memory")
    if selected is None:
        return None
    path = Path(settings.REPORT_DIRECTORY) / "Working_Memory_Report.pdf"
    return write_task_report(selected, "working_memory", "Working Memory", path)

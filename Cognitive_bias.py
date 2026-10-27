"""Generate the Cognitive Bias report."""
from pathlib import Path
import settings
from report_utils import enabled_task_data, write_task_report


def generate(data):
    selected = enabled_task_data(data, "cognitive_bias", "Cognitive_bias")
    if selected is None:
        return None
    path = Path(settings.REPORT_DIRECTORY) / "Cognitive_Bias_Report.pdf"
    return write_task_report(selected, "cognitive_bias", "Cognitive Bias", path)

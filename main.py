"""Prepare the trial dataset and generate the configured report files."""
from __future__ import annotations

import logging
from pathlib import Path

import Period_Comparison
import Status_Activity_Report
import Trials_Per_Task
import Weekly_Daily_Trials
import settings
from prepare_data import prepare_data

LOGGER = logging.getLogger("reports_nas")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = prepare_data(apply_filters=False)
    output_dir = Path(settings.OUTPUT_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = output_dir / settings.OUTPUT_FILENAME
    data.to_csv(prepared_path, sep=settings.CSV_SEPARATOR, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved prepared trial dataset: %s (%s rows)", prepared_path, len(data))

    # The combined report contains current status, recent daily trials,
    # trial-hour occupancy, and the configured Cognitive Bias plots.
    outputs = [Status_Activity_Report.generate(data)]
    trials_pdf, trials_csv = Trials_Per_Task.generate(data)
    period_workbook = Period_Comparison.generate(data)
    weekly_pdf, weekly_csv = Weekly_Daily_Trials.generate(data)
    outputs.extend([trials_pdf, trials_csv, period_workbook, weekly_pdf, weekly_csv])
    for output in outputs:
        if output is not None:
            LOGGER.info("Generated %s", output)


if __name__ == "__main__":
    main()

"""Prepare the trial dataset and generate the configured report files."""
from __future__ import annotations

import logging
from pathlib import Path

import Cognitive_bias
import Early_Training
import Probability
import Working_Memory_Report
import Period_Comparison
import Status_Activity_Report
import Trials_Per_Task
import Weekly_Daily_Trials
import settings
from prepare_data import prepare_data

LOGGER = logging.getLogger("reports_nas")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("Starting data preparation and environment merge...", flush=True)
    data = prepare_data(apply_filters=False)
    print(f"Data preparation complete: {len(data):,} trial rows.", flush=True)

    output_dir = Path(settings.OUTPUT_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = output_dir / settings.OUTPUT_FILENAME
    data.to_csv(prepared_path, sep=settings.CSV_SEPARATOR, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved prepared trial dataset: %s (%s rows)", prepared_path, len(data))
    print(f"Saved prepared trial data to: {prepared_path}", flush=True)

    outputs = []

    print("Generating Early Training report...", flush=True)
    outputs.append(Early_Training.generate(data))

    print("Generating Working Memory report...", flush=True)
    outputs.append(Working_Memory_Report.generate(data))

    print("Generating Cognitive Bias report...", flush=True)
    outputs.append(Cognitive_bias.generate(data))

    print("Generating Probability report...", flush=True)
    outputs.append(Probability.generate(data))

    print("Generating Status and Activity report...", flush=True)
    outputs.append(Status_Activity_Report.generate(data))

    print("Generating Trials per Task report...", flush=True)
    trials_pdf, trials_csv = Trials_Per_Task.generate(data)
    outputs.extend([trials_pdf, trials_csv])

    print("Generating period comparison workbook...", flush=True)
    outputs.append(Period_Comparison.generate(data))

    print("Generating weekly daily-trial reports...", flush=True)
    weekly_pdf, weekly_csv = Weekly_Daily_Trials.generate(data)
    outputs.extend([weekly_pdf, weekly_csv])

    for output in outputs:
        if output is not None:
            LOGGER.info("Generated %s", output)

    print("All reports finished.", flush=True)


if __name__ == "__main__":
    main()

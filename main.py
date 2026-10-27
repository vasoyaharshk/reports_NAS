"""Run sorting and then generate the configured report files."""
from __future__ import annotations

import logging
from pathlib import Path

import Cognitive_bias
import Early_Training
import Probability
import Trials_Per_Task
import Working_Memory_Report
import settings
from data_sorting import sort_trial_data

LOGGER = logging.getLogger("reports_nas")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = sort_trial_data(apply_filters=False)
    output_dir = Path(settings.OUTPUT_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_path = output_dir / settings.OUTPUT_FILENAME
    data.to_csv(sorted_path, sep=settings.CSV_SEPARATOR, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved complete sorted dataset: %s (%s rows)", sorted_path, len(data))

    # Each task module applies its own True/False setting. Early training and
    # trials per task are separate reports; trials per task sees every row.
    outputs = [
        Early_Training.generate(data),
        Working_Memory_Report.generate(data),
        Cognitive_bias.generate(data),
        Probability.generate(data),
    ]
    trials_pdf, trials_csv = Trials_Per_Task.generate(data)
    outputs.extend([trials_pdf, trials_csv])
    for output in outputs:
        if output is not None:
            LOGGER.info("Generated %s", output)


if __name__ == "__main__":
    main()

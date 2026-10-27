# reports_NAS

Python project for preparing Rat Village trial data for daily reports.

## First step: sort and combine trial data

1. Edit the six source paths in `settings.py`. Each village needs its `early_training` CSV and `global_trials_pi` CSV. The CSV files are read with the semicolon separator used by the R workflow.
2. Set `Working_Memory`, `Probability`, and `Cognitive_bias` to `True` or `False`. Enabled task families are retained in the sorted output. Working memory is configured for RV1 only; its source is loaded when `Working_Memory = True`. Add the actual RV1 file path in `settings.py` before enabling it.
3. Review `EXCLUDED_SUBJECTS`. Its initial values mirror the current R sorting script and can be changed for this reporting workflow.
4. Optionally set `INACTIVE_HOURS_FILE` to the workbook path if it contains the three Rat Village sheets.
5. Set `OUTPUT_DIRECTORY` to a writable NAS folder.
6. Install the dependencies and run:

```bash
python -m pip install pandas openpyxl
python data_sorting.py
```

The script writes a semicolon-delimited `sorted_trials.csv`. It adds village and source labels, a task category, parsed trial timestamps, response-side fields, success, condition type, session timing, and 60-trial Cognitive Bias block fields when their source columns are available. The 2025 Probability Weber's Law records are labeled as pre-test and other records as pilot. Task names outside the three configured families are retained by default and labeled `unclassified`.

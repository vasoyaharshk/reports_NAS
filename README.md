# reports_NAS

Python project for sorting Rat Village data and generating the daily report files.

## Configure

1. Edit the six source CSV paths in `settings.py`: `early_training` and `global_trials_pi` for each village. CSV files use the semicolon separator.
2. Set `Working_Memory`, `Probability`, and `Cognitive_bias` to `True` or `False`. These switches enable or skip their task-specific PDFs. Working Memory is collected only in RV1; set its RV1 path before enabling it.
3. Review `EXCLUDED_SUBJECTS` and optionally set `INACTIVE_HOURS_FILE`.
4. Set `EVENT_FILES` and `ENVIRONMENT_FILES` in `settings.py`. Add every file under its village number; event filenames do not need a village suffix. Each village value can be one path or a list of paths. Environment workbooks must have an `Environment` sheet with a `timestamp` column.
5. Set `OUTPUT_DIRECTORY` and `REPORT_DIRECTORY` to writable NAS folders.

## Run the full pipeline

Install dependencies and run the main entry point:

```bash
python -m pip install pandas openpyxl matplotlib
python main.py
```

`main.py` sorts the complete dataset, adds configured environment readings and event-derived session timings, saves the enriched all-column dataset, then calls the early-training, Working Memory, Cognitive Bias, Probability, trials-per-task, and period-comparison modules in sequence.

The task switches affect only the three task-specific reports. `Trials_Per_Task.py` and `Period_Comparison.py` always use the complete sorted data, regardless of those switches. The trials module writes `Trials_Per_Task_Report.pdf` and `Trials_Per_Task_Daily.csv`. The period-comparison module writes `RV2_RV3_Trials_Sessions_By_Period.xlsx`, with village and subject comparison sheets. Its periods, session ID column, and timezone are configurable in `settings.py`; period boundaries use Europe/Madrid time by default. PDFs and workbooks are saved in `REPORT_DIRECTORY`.



`data_enrichment.py` reads all configured Academy event files using their explicit village mapping, so event filenames may be renamed. It matches animal `START` and `END` records by village, subject, task label, and time. Academy event timestamps are interpreted as local Europe/Madrid wall time, matching the repository's `datetime.now()` event writer; no one-hour adjustment is applied. If an end event is missing, the start time is still attached and the missing end remains blank. Environment readings are matched to the latest reading at or before each trial, within the configured 15-second maximum gap. The input files are read only.

The task reports currently use a shared layout: a task overview page followed by village pages showing valid trials per session and mean session accuracy. This gives each category its own module and PDF so that its plots can be refined independently. The attached R file does not contain a Working Memory plotting section, so the first Working Memory PDF uses the shared summary layout and the configured Working Memory task rows.

## Run sorting only

```bash
python data_sorting.py
```

This writes the full semicolon-delimited `sorted_trials.csv`. It merges the source files while retaining the union of their original columns, then adds derived fields for reporting. Original fields such as `correct_th` and `incorrect_th` remain present alongside the derived position fields. Unlike the R sorter, the Python pipeline does not project the datasheet down to a selected column list. It parses timestamps, derives response and session fields, applies historical task-name corrections, recalculates 60-trial Cognitive Bias blocks, and applies the RV2 half-screen correction from the existing R workflow. Unknown task names remain available in the complete data and are marked `unclassified`.

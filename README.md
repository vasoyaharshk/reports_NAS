# reports_NAS

Python project for preparing Rat Village data and generating the daily report files.

## Configure

1. Set `DATA_DIRECTORY` in `settings.py` to the folder shown as `H:\NAS`. The script reads `early_training*.csv`, `global*.csv`, and RV1 `global_trials_wm*.csv` from each village's `global_trials_rvN` folder. CSV files use the semicolon separator.
2. Set `Working_Memory`, `Probability`, and `Cognitive_bias` to `True` or `False`. These switches enable or skip their task-specific PDFs. Working Memory is collected only in RV1; set `Working_Memory = True` to include its report.
3. Review `EXCLUDED_SUBJECTS` and optionally set `INACTIVE_HOURS_FILE`.
4. The settings point to `events_rv1`, `events_rv2`, and `events_rv3`, and to `environment_rv2` and `environment_rv3`. Every `events*.csv` and supported `environment*` CSV/Excel file in those folders is loaded. RV1 has no environment folder configured.
5. Set `OUTPUT_DIRECTORY` and `REPORT_DIRECTORY` to writable NAS folders.

## Run the full pipeline

Install dependencies and run the main entry point:

```bash
python -m pip install pandas openpyxl matplotlib
python main.py
```

During preparation, environment exports are merged per village into `H:\NAS\\environment\\environment_rv2.csv` and `environment_rv3.csv`. CSV and Excel files are supported. Columns are aligned by name, timestamps are validated and sorted, and exact duplicate rows are removed. Source files are kept and source folders are never written to. Rows with the same timestamp but different readings are retained.

`main.py` is the master script. Run `python main.py` to prepare the all-column trial dataset and generate Early Training, Working Memory (when enabled), Cognitive Bias, Probability, Status and Activity, Trials per Task, weekly daily-trial reports, and period-comparison reports. Prepared files go to the repository's `output` and `reports` folders.

The task switches affect only the three task-specific reports. `Trials_Per_Task.py` and `Period_Comparison.py` always use the complete sorted data, regardless of those switches. The trials module writes `Trials_Per_Task_Report.pdf` and `Trials_Per_Task_Daily.csv`. The period-comparison module writes `RV2_RV3_Trials_Sessions_By_Period.xlsx`, with village and subject comparison sheets. Its periods, session ID column, and timezone are configurable in `settings.py`; period boundaries use Europe/Madrid time by default. PDFs and workbooks are saved in `REPORT_DIRECTORY`.

`prepare_data.py` contains the data preparation functions. It reads configured Academy event files by village, matches animal `START` and `END` records by village, subject, task label, and time, and interprets Academy event timestamps as fixed UTC+1 wall time before converting to UTC. If an end event is missing, the start is still attached and the end remains blank. Event temperature/humidity and the nearest environment reading are stored once per session. All sensor columns ending in `_C` or `_RH` are included except the `control_C` setpoint. The selected environment timestamp and its signed offset from the session start are saved as `env_timestamp` and `env_time_from_session_start_seconds`. Session values are carried onto that session's trial rows.

The task reports currently use a shared layout: a task overview page followed by village pages showing valid trials per session and mean session accuracy. This gives each category its own module and PDF so that its plots can be refined independently. The attached R file does not contain a Working Memory plotting section, so the first Working Memory PDF uses the shared summary layout and the configured Working Memory task rows.

## Prepare the data only

```bash
python prepare_data.py
```

This writes the full semicolon-delimited `prepared_trial_data.csv`. It prepares the source files, parses timestamps, derives response and session fields, applies historical task-name corrections, recalculates 60-trial Cognitive Bias blocks, and applies the RV2 half-screen correction from the existing R workflow. Unknown task names remain available in the complete data and are marked `unclassified`.

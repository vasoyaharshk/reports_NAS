# reports_NAS

Python project for preparing Rat Village data and generating the daily report files.

## Configure

1. Edit the six source CSV paths in `settings.py`: `early_training` and `global_trials_pi` for each village. CSV files use the semicolon separator.
2. Set `Working_Memory`, `Probability`, and `Cognitive_bias` to `True` or `False`. These switches enable or skip their task-specific PDFs. Working Memory is collected only in RV1; set its RV1 path before enabling it.
3. Review `EXCLUDED_SUBJECTS` and optionally set `INACTIVE_HOURS_FILE`.
4. Set `EVENT_FILES` in `settings.py`. Add every event file under its village number; event filenames do not need a village suffix. Each village value can be one path or a list of paths. Set `ENVIRONMENT_DIRECTORIES` to the separate environment folder for each village.
5. Set `OUTPUT_DIRECTORY` and `REPORT_DIRECTORY` to writable NAS folders.

## Run the full pipeline

Install dependencies and run the main entry point:

```bash
python -m pip install pandas openpyxl matplotlib
python main.py
```

At the start of data preparation, environment exports are merged separately for each village into `environment_rvN_YYYYMMDD_HHMMSS.csv`. CSV and Excel files are supported. Columns are combined by name within each village, so new sensor columns are retained and older files receive blank cells for sensors they did not record. Timestamps and the saved output are validated before source exports are deleted. Duplicate rows and repeated timestamps are retained. RV2 and RV3 keep separate sensor columns. Set `DELETE_ENVIRONMENT_INPUTS_AFTER_MERGE` in `settings.py` to control source deletion.

`main.py` calls `prepare_data.py` to combine and prepare the trial files, merge environment exports, add session start/end times from Academy events, and attach the nearest environment sensor sample for each session start. It saves the enriched all-column dataset, then calls the early-training, Working Memory, Cognitive Bias, Probability, trials-per-task, and period-comparison modules in sequence.

The task switches affect only the three task-specific reports. `Trials_Per_Task.py` and `Period_Comparison.py` always use the complete sorted data, regardless of those switches. The trials module writes `Trials_Per_Task_Report.pdf` and `Trials_Per_Task_Daily.csv`. The period-comparison module writes `RV2_RV3_Trials_Sessions_By_Period.xlsx`, with village and subject comparison sheets. Its periods, session ID column, and timezone are configurable in `settings.py`; period boundaries use Europe/Madrid time by default. PDFs and workbooks are saved in `REPORT_DIRECTORY`.

`prepare_data.py` contains the data preparation functions. It reads configured Academy event files by village, matches animal `START` and `END` records by village, subject, task label, and time, and interprets Academy timestamps as Europe/Madrid wall time before converting to UTC. If an end event is missing, the start is still attached and the end remains blank. Event temperature/humidity and the nearest environment reading are stored once per session. All sensor columns ending in `_C` or `_RH` are included except the `control_C` setpoint. The selected environment timestamp and its signed offset from the session start are saved as `env_timestamp` and `env_time_from_session_start_seconds`. Session values are carried onto that session's trial rows.

The task reports currently use a shared layout: a task overview page followed by village pages showing valid trials per session and mean session accuracy. This gives each category its own module and PDF so that its plots can be refined independently. The attached R file does not contain a Working Memory plotting section, so the first Working Memory PDF uses the shared summary layout and the configured Working Memory task rows.

## Prepare the data only

```bash
python prepare_data.py
```

This writes the full semicolon-delimited `sorted_trials.csv`. It prepares the source files, parses timestamps, derives response and session fields, applies historical task-name corrections, recalculates 60-trial Cognitive Bias blocks, and applies the RV2 half-screen correction from the existing R workflow. Unknown task names remain available in the complete data and are marked `unclassified`.

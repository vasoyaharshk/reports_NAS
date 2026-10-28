# reports_NAS

Python project for sorting Rat Village data and generating the daily report files.

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

## Merge environment exports

After all source exports for a village are in its configured environment folder, merge them with:

```bash
python merge_environment_files.py
```

The script merges each village separately and writes `environment_rvN_YYYYMMDD_HHMMSS.csv` in that village's folder. It accepts semicolon or comma-delimited CSV files and Excel workbooks using the configured `Environment` sheet. It requires matching column schemas within each village, validates every timestamp, sorts rows by timestamp, and verifies an exact CSV round-trip and row count before deleting any inputs. Duplicate rows and repeated timestamps are retained. RV2 and RV3 outputs keep their own sensor columns. Input deletion is controlled by `DELETE_ENVIRONMENT_INPUTS_AFTER_MERGE`.

`main.py` sorts the complete dataset, adds event-derived session timings and session-level temperature/humidity values, saves the enriched all-column dataset, then calls the early-training, Working Memory, Cognitive Bias, Probability, trials-per-task, and period-comparison modules in sequence.

The task switches affect only the three task-specific reports. `Trials_Per_Task.py` and `Period_Comparison.py` always use the complete sorted data, regardless of those switches. The trials module writes `Trials_Per_Task_Report.pdf` and `Trials_Per_Task_Daily.csv`. The period-comparison module writes `RV2_RV3_Trials_Sessions_By_Period.xlsx`, with village and subject comparison sheets. Its periods, session ID column, and timezone are configurable in `settings.py`; period boundaries use Europe/Madrid time by default. PDFs and workbooks are saved in `REPORT_DIRECTORY`.

`data_enrichment.py` reads all configured Academy event files using their explicit village mapping, so event filenames may be renamed. It matches animal `START` and `END` records by village, subject, task label, and time. Academy event timestamps are interpreted as local Europe/Madrid wall time, matching the repository's `datetime.now()` event writer; no one-hour adjustment is applied. If an end event is missing, the start time is still attached and the missing end remains blank. Temperature and humidity are taken independently from the first matching Academy `ACTION` event after the session starts, within that session window, and are then associated with the session's trial rows. This avoids requiring both sensor values to appear in one event. The event files are read only. Environment workbook and CSV data is consolidated separately by `merge_environment_files.py`; it is not yet matched to trials.

The task reports currently use a shared layout: a task overview page followed by village pages showing valid trials per session and mean session accuracy. This gives each category its own module and PDF so that its plots can be refined independently. The attached R file does not contain a Working Memory plotting section, so the first Working Memory PDF uses the shared summary layout and the configured Working Memory task rows.

## Run sorting only

```bash
python data_sorting.py
```

This writes the full semicolon-delimited `sorted_trials.csv`. It merges the source files, parses timestamps, derives response and session fields, applies historical task-name corrections, recalculates 60-trial Cognitive Bias blocks, and applies the RV2 half-screen correction from the existing R workflow. Unknown task names remain available in the complete data and are marked `unclassified`.

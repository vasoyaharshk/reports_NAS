"""Prepare the complete trial dataset, session events, and environment readings."""
from __future__ import annotations

import csv
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import settings

LOGGER = logging.getLogger("reports_nas.prepare_data")
SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}
MERGED_NAME_PATTERN = re.compile(r"^environment_rv[123]_\d{8}_\d{6}\.csv$", re.I)
ENVIRONMENT_SESSION_COLUMNS = (
    "env_timestamp", "env_time_from_session_start_seconds", "env_match_status",
)

def _read_source(path: Path, village: int, dataset: str) -> pd.DataFrame:
    """Read a source CSV without imposing incompatible column types."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Input file not found for RV{village} ({dataset}): {path}. "
            "Update VILLAGE_INPUTS in settings.py."
        )
    frame = pd.read_csv(
        path,
        sep=settings.CSV_SEPARATOR,
        low_memory=False,
        na_values=["nan", "NaN", "NaN"],
        keep_default_na=True,
    )
    frame["ratvillage"] = village
    frame["dataset"] = f"{dataset}_rv{village}"
    return frame


def load_all_villages() -> pd.DataFrame:
    """Load the early-training and global-trial exports for RV1–RV3."""
    frames: list[pd.DataFrame] = []
    expected = {"early_training", "global_trials_pi"}
    for village in (1, 2, 3):
        inputs = settings.VILLAGE_INPUTS.get(village)
        if not inputs or not expected.issubset(inputs):
            raise ValueError(
                f"VILLAGE_INPUTS[{village}] must define {sorted(expected)}."
            )
        for dataset in ("early_training", "global_trials_pi"):
            frames.append(_read_source(Path(inputs[dataset]), village, dataset))

    # Include RV1 Working Memory whenever its source exists, even if the
    # Working_Memory report is switched off. This keeps the independent
    # trials-per-task output complete. Enabling the report requires the source.
    wm_path = settings.VILLAGE_INPUTS.get(1, {}).get("working_memory")
    if wm_path is not None and Path(wm_path).is_file():
        frames.append(_read_source(Path(wm_path), 1, "working_memory"))
    elif settings.Working_Memory:
        raise ValueError(
            "Working_Memory is enabled, but its RV1 source is missing. Set the working_memory path in settings.py."
        )

    # sort=False preserves source column order and adds columns present in other files.
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def _parse_epoch_or_datetime(values: pd.Series) -> pd.Series:
    """Parse Unix seconds or datetime strings into UTC timestamps."""
    numeric = pd.to_numeric(values, errors="coerce")
    is_numeric = numeric.notna()
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    if is_numeric.any():
        result.loc[is_numeric] = pd.to_datetime(
            numeric.loc[is_numeric], unit="s", origin="unix", utc=True, errors="coerce"
        )
    if (~is_numeric).any():
        result.loc[~is_numeric] = pd.to_datetime(
            values.loc[~is_numeric], utc=True, errors="coerce", format="mixed"
        )
    return result


def _task_family(task: Any) -> str:
    name = re.sub(r"[^a-z0-9]+", "", str(task).strip().lower())
    if not name or name == "nan" or name == "none":
        return "unclassified"
    if "cognitivebias" in name or name.startswith("cb"):
        return "cognitive_bias"
    if "workingmemory" in name or name.startswith("wm"):
        return "working_memory"
    if "probability" in name or "weber" in name or name.startswith("pi"):
        return "probability"
    return "unclassified"


def apply_task_switches(data: pd.DataFrame) -> pd.DataFrame:
    """Keep task rows enabled in settings.py; unknown task names remain visible."""
    if "task" not in data.columns:
        LOGGER.warning("No 'task' column found; task switches cannot be applied.")
        return data

    enabled = {
        "working_memory": bool(settings.Working_Memory),
        "probability": bool(settings.Probability),
        "cognitive_bias": bool(settings.Cognitive_bias),
    }
    data = data.copy()
    data["task_category"] = data["task"].map(_task_family)
    keep = data["task_category"].map(
        lambda category: (
            bool(settings.INCLUDE_UNCLASSIFIED_TASKS)
            if category == "unclassified"
            else enabled[category]
        )
    )
    LOGGER.info("Task switches: %s", enabled)
    LOGGER.info("Rows by category before filtering: %s", data["task_category"].value_counts(dropna=False).to_dict())
    filtered = data.loc[keep].copy()
    LOGGER.info("Rows retained after task filtering: %s", len(filtered))
    return filtered


def _add_core_fields(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the common cleaning and derived columns from the R workflow."""
    data = data.copy()

    if "correct_th" in data.columns and "incorrect_th" in data.columns:
        # Keep the source fields intact and add the aliases used by downstream
        # report calculations. The final sorted datasheet retains every input
        # column from the union of source files.
        if "correct_th_pos" not in data.columns:
            data["correct_th_pos"] = data["correct_th"]
        if "incorrect_th_pos" not in data.columns:
            data["incorrect_th_pos"] = data["incorrect_th"]

    if "subject" in data.columns:
        data["subject"] = data["subject"].astype("string").str.strip()
        excluded = {str(subject).lower() for subject in settings.EXCLUDED_SUBJECTS}
        data = data.loc[~data["subject"].str.lower().isin(excluded)].copy()

    if "TRIAL_START" in data.columns:
        data["datetime"] = _parse_epoch_or_datetime(data["TRIAL_START"])
    elif "datetime" in data.columns:
        data["datetime"] = _parse_epoch_or_datetime(data["datetime"])

    if "TRIAL_END" in data.columns:
        data["datetime_end"] = _parse_epoch_or_datetime(data["TRIAL_END"])
    elif "datetime_end" in data.columns:
        data["datetime_end"] = _parse_epoch_or_datetime(data["datetime_end"])

    if "datetime" in data.columns:
        # Match the day/night split in the R sorting workflow (08:00–20:00 UTC).
        hour = data["datetime"].dt.hour
        data["day_night"] = pd.Series("night", index=data.index, dtype="string")
        data.loc[hour.between(8, 19), "day_night"] = "day"

    if {"stage", "substage"}.issubset(data.columns):
        valid = data["stage"].notna() & data["substage"].notna()
        data["stage_substage"] = pd.NA
        data.loc[valid, "stage_substage"] = (
            data.loc[valid, "stage"].astype(str) + "." + data.loc[valid, "substage"].astype(str)
        )

    if {"stage", "substage"}.issubset(data.columns):
        valid = data["stage"].notna() & data["substage"].notna()
        data["stage_substage_bias"] = pd.NA
        values = data.loc[valid, "stage"].astype(str) + "." + data.loc[valid, "substage"].astype(str)
        if "substage_bias" in data.columns:
            bias_valid = valid & data["substage_bias"].notna()
            data.loc[valid, "stage_substage_bias"] = values
            data.loc[bias_valid, "stage_substage_bias"] = (
                data.loc[bias_valid, "stage"].astype(str)
                + "." + data.loc[bias_valid, "substage"].astype(str)
                + "." + data.loc[bias_valid, "substage_bias"].astype(str)
            )
        else:
            data.loc[valid, "stage_substage_bias"] = values

    if "trial_result" in data.columns:
        result = data["trial_result"].astype("string").str.lower()
        data["success"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
        valid = result.notna()
        data.loc[valid, "success"] = result.loc[valid].isin(["correct_first", "correct"]).astype(int)

    if "date" in data.columns:
        # Preserve date values as dates; epoch-based date columns are also supported.
        numeric_date = pd.to_numeric(data["date"], errors="coerce")
        parsed_date = pd.to_datetime(data["date"], errors="coerce", format="mixed")
        epoch_mask = numeric_date.notna()
        if epoch_mask.any():
            parsed_date.loc[epoch_mask] = pd.to_datetime(
                numeric_date.loc[epoch_mask], unit="s", origin="unix", errors="coerce"
            )
        data["date"] = parsed_date.dt.date

    if {"task", "date"}.issubset(data.columns):
        years = pd.to_datetime(data["date"], errors="coerce").dt.year
        is_weber = data["task"].astype("string").eq("Probability_WebersLaw")
        data.loc[is_weber & years.eq(2025), "task"] = "Probability_WebersLaw_Pre"
        data.loc[is_weber & years.ne(2025), "task"] = "Probability_WebersLaw_Pilot"

    if {"current_condition", "trial_condition"}.intersection(data.columns):
        current = pd.to_numeric(data.get("current_condition", pd.Series(pd.NA, index=data.index)), errors="coerce")
        trial = pd.to_numeric(data.get("trial_condition", pd.Series(pd.NA, index=data.index)), errors="coerce")
        condition = current.combine_first(trial)
        data["condition_type"] = pd.Series(pd.NA, index=data.index, dtype="string")
        valid = condition.notna()
        data.loc[valid & condition.mod(2).eq(0), "condition_type"] = "big"
        data.loc[valid & condition.mod(2).eq(1), "condition_type"] = "small"

    return data


def _rename_historical_tasks(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the subject- and date-specific run labels used in the R workflow."""
    required = {"subject", "task", "date"}
    if not required.issubset(data.columns):
        return data
    data = data.copy()
    date = pd.to_datetime(data["date"], errors="coerce")
    subject = data["subject"].astype("string").str.lower()
    task = data["task"].astype("string")

    def rename(mask: pd.Series, new_name: str) -> None:
        data.loc[mask.fillna(False), "task"] = new_name

    before = date.lt(pd.Timestamp("2025-07-10"))
    rename(subject.eq("sorrel") & before & task.eq("Probability_Training_BB_Size_Acc"), "Probability_Training_BB_Size_Acc_First_Run")
    rename(subject.eq("sorrel") & date.ge(pd.Timestamp("2025-07-10")) & task.eq("Probability_WebersLaw_Pre"), "Probability_WebersLaw_Pre_Second_Run")
    rename(subject.eq("sorrel") & before & task.eq("Probability_WL_Training_Acc"), "Probability_WL_Training_Acc_First_Run")

    before = date.lt(pd.Timestamp("2025-08-05"))
    rename(subject.eq("ciri") & before & task.eq("Probability_Training_BB_Size_Acc"), "Probability_Training_BB_Size_Acc_First_Run")
    rename(subject.eq("ciri") & date.ge(pd.Timestamp("2025-08-05")) & task.eq("Probability_WebersLaw_Pre"), "Probability_WebersLaw_Pre_Second_Run")
    rename(subject.eq("ciri") & before & task.eq("Probability_WL_Training_Acc"), "Probability_WL_Training_Acc_First_Run")

    rename(subject.eq("gal") & date.lt(pd.Timestamp("2025-08-27")) & task.eq("Probability_WL_Training_Runthrough_Acc"), "Probability_WL_Training_Runthrough_Acc_First_Run")
    rename(subject.eq("gal") & date.ge(pd.Timestamp("2025-08-27")) & date.lt(pd.Timestamp("2025-11-30")) & task.eq("Probability_WL_Training_Runthrough_Acc"), "Probability_WL_Training_Runthrough_Acc_Second_Run")
    rename(subject.eq("gal") & date.ge(pd.Timestamp("2025-08-27")) & task.eq("Probability_WebersLaw_Pre"), "Probability_WebersLaw_Pre_Second_Run")
    rename(subject.eq("gal") & date.lt(pd.Timestamp("2025-08-27")) & task.eq("Probability_WL_Training_Acc"), "Probability_WL_Training_Acc_First_Run")
    rename(subject.eq("gal") & date.lt(pd.Timestamp("2025-08-27")) & task.eq("Probability_WebersLaw_Post"), "Probability_WebersLaw_Post_First_Run")

    rename(subject.eq("monika") & date.lt(pd.Timestamp("2025-12-03")) & task.eq("Probability_Extra_Training_Acc_Stage_2"), "Probability_Extra_Training_Acc_Stage_2_First_Run")
    rename(subject.eq("monika") & date.lt(pd.Timestamp("2025-12-03")) & task.eq("Probability_Extra_Training_Acc"), "Probability_Extra_Training_Acc_First_Run")
    rename(subject.eq("phoebe") & date.lt(pd.Timestamp("2025-12-09")) & task.eq("Probability_Extra_Training_Acc_Stage_2"), "Probability_Extra_Training_Acc_Stage_2_First_Run")
    rename(subject.eq("phoebe") & date.lt(pd.Timestamp("2025-12-09")) & task.eq("Probability_Extra_Training_Acc"), "Probability_Extra_Training_Acc_First_Run")
    rename(subject.eq("phoebe") & date.lt(pd.Timestamp("2026-01-25")) & task.eq("A_TouchTeaching_blue"), "A_TouchTeaching_blue_First_Run")
    rename(subject.eq("phoebe") & date.lt(pd.Timestamp("2026-01-30")) & task.eq("Probability_Extra_Training_Acc_FF"), "Probability_Extra_Training_Acc_FF_First_Run")
    rename(subject.eq("joy") & date.ge(pd.Timestamp("2025-03-11")) & task.eq("Probability_WebersLaw_Pre"), "Probability_WebersLaw_Pre_Second_Run")
    rename(subject.isin(["gal", "sparky"]) & date.lt(pd.Timestamp("2026-05-29")) & task.eq("Probability_Training_BT_Acc_FF"), "Probability_Training_BT_Acc_FF_First_Run")
    return data


def _add_response_fields(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "response_x" in data.columns and "trial_result" in data.columns:
        def last_response(value: Any) -> float:
            if pd.isna(value):
                return float("nan")
            try:
                return float(str(value).split(",")[-1].strip())
            except (TypeError, ValueError):
                return float("nan")

        data["lastresponse"] = data["response_x"].map(last_response)
        is_miss = data["trial_result"].astype("string").str.lower().eq("miss")
        data.loc[is_miss, "lastresponse"] = float("nan")
        data["response_side"] = "miss"
        valid = data["lastresponse"].notna() & ~is_miss
        data.loc[valid & (data["lastresponse"] < 200), "response_side"] = "left"
        data.loc[valid & (data["lastresponse"] >= 200), "response_side"] = "right"

    correct_column = "correct_th_pos" if "correct_th_pos" in data.columns else "correct_th"
    grouping = [column for column in ("ratvillage", "subject", "task", "stage_substage", "session") if column in data.columns]
    if correct_column in data.columns and grouping:
        numeric = pd.to_numeric(data[correct_column], errors="coerce")
        grouped = data.assign(_correct_numeric=numeric).groupby(grouping, dropna=False)["_correct_numeric"]
        low = grouped.transform("min")
        high = grouped.transform("max")
        data[correct_column] = pd.Series(pd.NA, index=data.index, dtype="string")
        data.loc[numeric.notna() & numeric.eq(low), correct_column] = "left"
        data.loc[numeric.notna() & numeric.eq(high) & numeric.ne(low), correct_column] = "right"
    return data


def _add_session_fields(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    group_columns = [column for column in ("ratvillage", "subject", "session") if column in data.columns]
    if {"subject", "session", "datetime", "datetime_end"}.issubset(data.columns):
        grouped = data.dropna(subset=["subject", "session"]).groupby(group_columns, dropna=False)
        starts = grouped["datetime"].transform("min")
        ends = grouped["datetime_end"].transform("max")
        duration = (ends - starts).dt.total_seconds() / 60
        data["session_start"] = starts.reindex(data.index)
        data["session_end"] = ends.reindex(data.index)
        data["session_duration"] = duration.reindex(data.index)
        data["seconds_from_start"] = (data["datetime"] - data["session_start"]).dt.total_seconds()
        data["minutes_from_start"] = (data["seconds_from_start"] // 60).astype("Int64")
        data["datetime_h"] = data["datetime"].dt.strftime("%Y-%m-%d %H:%M")
        data["datetime_end_h"] = data["datetime_end"].dt.strftime("%Y-%m-%d %H:%M")

    if "tone_played" in data.columns:
        data["tone_played"] = data["tone_played"].astype("string").str.lower().str.title()

    required = {"task", "forced_choice_actual_trial", "subject", "session"}
    if required.issubset(data.columns):
        data["forced_choice_actual_trial"] = pd.to_numeric(data["forced_choice_actual_trial"], errors="coerce")
        group_columns = [column for column in ("ratvillage", "subject", "group", "pair", "task") if column in data.columns]
        for source_task, output_task in (
            ("Cognitive_Bias_Auditory_Training", "Cognitive_Bias_Auditory_Training_60R"),
            ("Cognitive_Bias_Auditory_Training_PR", "Cognitive_Bias_Auditory_Training_PR_60R"),
        ):
            selected = data["task"].eq(source_task)
            free_choice = selected & data["forced_choice_actual_trial"].eq(0)
            if not free_choice.any():
                continue
            ordered = data.loc[free_choice]
            sort_columns = group_columns + (["datetime"] if "datetime" in data.columns else [])
            order = ordered.sort_values(sort_columns, kind="stable", na_position="last").index
            block_index = data.loc[order].groupby(group_columns, dropna=False).cumcount() + 1
            data.loc[order, "block_number"] = ((block_index - 1) // 60 + 1).to_numpy()
            data.loc[order, "block_trial_counter"] = ((block_index - 1) % 60 + 1).to_numpy()
            data.loc[selected, "block_size"] = 60
            data.loc[selected, "task"] = output_task
    return data


def _apply_cb_half_screen_correction(data: pd.DataFrame) -> pd.DataFrame:
    """Recalculate RV2 Cognitive Bias responses recorded before the screen change."""
    required = {"ratvillage", "datetime", "task", "trial_result", "response_x", "side"}
    if not required.issubset(data.columns):
        return data
    data = data.copy()
    cutoff = pd.Timestamp("2026-04-30 16:30:00", tz="Europe/Madrid").tz_convert("UTC")
    cb_tasks = {
        "Cognitive_Bias_Auditory_Training_60R",
        "Cognitive_Bias_Auditory_Training_60",
        "Cognitive_Bias_Auditory_Training_PR_60R",
        "Cognitive_Bias_Auditory_Training_60_PR",
    }
    use = data["ratvillage"].eq(2) & data["datetime"].lt(cutoff) & data["task"].isin(cb_tasks)
    if not use.any():
        return data

    for column in ("success", "trial_result", "reward"):
        if column in data.columns:
            data[f"{column}_original"] = data[column]

    response = pd.to_numeric(data["response_x"], errors="coerce")
    response_side = pd.Series(pd.NA, index=data.index, dtype="string")
    response_side.loc[use & response.lt(205)] = "left"
    response_side.loc[use & response.ge(205)] = "right"
    correct_side = data["side"].astype("string").str.lower()
    is_miss = data["trial_result"].astype("string").str.lower().eq("miss")
    success = pd.Series(pd.NA, index=data.index, dtype="Float64")
    valid = use & ~is_miss & response_side.notna() & correct_side.notna()
    success.loc[valid] = response_side.loc[valid].eq(correct_side.loc[valid]).astype(int)

    trial_result = data["trial_result"].astype("string").copy()
    trial_result.loc[use & is_miss] = "miss"
    trial_result.loc[valid & success.eq(1)] = "correct"
    trial_result.loc[valid & success.eq(0)] = "incorrect"
    data["success_half"] = success
    data["trial_result_half"] = trial_result

    if "reward" in data.columns:
        reward = pd.to_numeric(data["reward"], errors="coerce").astype("Float64")
        unrewarded = pd.to_numeric(data.get("unrewarded_trial", pd.Series(pd.NA, index=data.index)), errors="coerce")
        valve = pd.to_numeric(data.get("valve_factor_c", pd.Series(pd.NA, index=data.index)), errors="coerce")
        reward_half = reward.copy()
        reward_half.loc[use & trial_result.eq("incorrect")] = 0
        reward_half.loc[use & trial_result.eq("miss")] = 0
        reward_half.loc[use & trial_result.eq("correct") & unrewarded.eq(0) & valve.eq(0.7)] = 70
        reward_half.loc[use & trial_result.eq("correct") & unrewarded.eq(0) & valve.eq(1.4)] = 140
        reward_half.loc[use & trial_result.eq("correct") & unrewarded.eq(1)] = 0
        data["reward_half"] = reward_half
        data["reward"] = reward_half
    data["success"] = data.get("success", pd.Series(pd.NA, index=data.index)).astype("Float64")
    data.loc[use, "success"] = success.loc[use]
    data["trial_result"] = data["trial_result"].astype("string")
    data.loc[use, "trial_result"] = trial_result.loc[use]
    return data


def _add_inactive_hours(data: pd.DataFrame) -> pd.DataFrame:
    workbook = settings.INACTIVE_HOURS_FILE
    if workbook is None:
        return data
    workbook = Path(workbook)
    if not workbook.is_file():
        raise FileNotFoundError(f"Inactive-hours workbook not found: {workbook}")

    pieces = []
    for village in (1, 2, 3):
        sheet = f"ratvillage{village}"
        inactive = pd.read_excel(workbook, sheet_name=sheet)
        if not {"date", "inactive_hours"}.issubset(inactive.columns):
            raise ValueError(f"Sheet {sheet} must contain date and inactive_hours columns.")
        inactive["ratvillage"] = village
        pieces.append(inactive[["ratvillage", "date", "inactive_hours"]])
    lookup = pd.concat(pieces, ignore_index=True)
    lookup["date"] = pd.to_datetime(lookup["date"], errors="coerce").dt.date
    lookup = lookup.drop_duplicates(["ratvillage", "date"], keep="last")
    merged = data.merge(lookup, on=["ratvillage", "date"], how="left", suffixes=("", "_lookup"))
    merged["inactive_hours"] = pd.to_numeric(merged["inactive_hours"], errors="coerce").fillna(0)
    merged["total_active_hours"] = 24 - merged["inactive_hours"]
    return merged


def _prepare_trial_table(apply_filters: bool = False) -> pd.DataFrame:
    """Merge and clean every source row.

    Task switches are report switches. Keeping this base dataset complete lets
    the independent trials-per-task report include disabled task families.
    Call with apply_filters=True only when a task-filtered export is wanted.
    """
    data = load_all_villages()
    if "trial_result" in data.columns:
        data = data.loc[data["trial_result"].notna()].copy()
    data = _add_core_fields(data)
    data = _rename_historical_tasks(data)
    data = _add_response_fields(data)
    data = _add_session_fields(data)
    data = _apply_cb_half_screen_correction(data)
    data = _add_inactive_hours(data)
    if "task" in data.columns:
        data["task_category"] = data["task"].map(_task_family)
    if apply_filters:
        data = apply_task_switches(data)

    if "datetime" in data.columns:
        data = data.sort_values(["ratvillage", "subject", "datetime"], kind="stable", na_position="last")
    return data.reset_index(drop=True)


def _discover_inputs(directory: Path, village: int) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Environment directory for RV{village} not found: {directory}")
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and path.name.lower().startswith("environment")
        and not path.name.startswith(".")
    )
    if not files:
        LOGGER.info("No environment files configured for RV%s in %s", village, directory)
    return files


def _read_csv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=";,\t,").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        delimiter = max((";", ",", "\t"), key=first_line.count)
    frame = pd.read_csv(
        path, sep=delimiter, dtype="string", keep_default_na=False,
        encoding="utf-8-sig", low_memory=False,
    )
    return frame


def _read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = _read_csv(path)
    else:
        frame = pd.read_excel(
            path, sheet_name=settings.ENVIRONMENT_EXCEL_SHEET, dtype=object
        )
    frame.columns = [str(column).strip().lstrip("\ufeff") for column in frame.columns]
    if not frame.columns.is_unique:
        raise ValueError(f"Duplicate column names after trimming in {path.name}.")
    if not len(frame.columns):
        raise ValueError(f"No columns found in environment file {path.name}.")

    # Use text values for a faithful CSV round-trip across CSV and Excel inputs.
    for column in frame.columns:
        frame[column] = frame[column].map(
            lambda value: "" if pd.isna(value) else str(value)
        ).astype("string")
    return frame


def _timestamp_column(frame: pd.DataFrame, source: Path) -> str:
    matches = [column for column in frame.columns if column.strip().lower() == "timestamp"]
    if len(matches) != 1:
        raise ValueError(f"{source.name} must contain exactly one 'timestamp' column.")
    return matches[0]


def _merged_output_path(directory: Path, village: int) -> Path:
    local_now = datetime.now(ZoneInfo("Europe/Madrid"))
    stamp = local_now.strftime("%Y%m%d_%H%M%S")
    output = directory / f"environment_rv{village}_{stamp}.csv"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing merged file: {output}")
    return output


def _prepare_merge(directory: Path, village: int) -> dict | None:
    inputs = _discover_inputs(directory, village)
    if not inputs:
        return None
    existing_merged = [path for path in inputs if MERGED_NAME_PATTERN.match(path.name)]
    if len(inputs) == 1 and existing_merged:
        LOGGER.info("RV%s already has one merged file; nothing to do.", village)
        return None

    frames: list[pd.DataFrame] = []
    timestamp_name: str | None = None
    union_columns: list[str] = []
    input_rows: dict[str, int] = {}

    for path in inputs:
        frame = _read_input(path)
        current_timestamp_name = _timestamp_column(frame, path)
        if timestamp_name is None:
            timestamp_name = current_timestamp_name
        elif current_timestamp_name != timestamp_name:
            if timestamp_name in frame.columns:
                raise ValueError(f"Ambiguous timestamp columns in {path.name}.")
            frame = frame.rename(columns={current_timestamp_name: timestamp_name})
            current_timestamp_name = timestamp_name
        for column in frame.columns:
            if column not in union_columns:
                union_columns.append(column)
        if "__merge_timestamp" in frame.columns:
            raise ValueError(f"Reserved helper column '__merge_timestamp' found in {path.name}.")
        frame["__merge_timestamp"] = pd.to_datetime(
            frame[current_timestamp_name], errors="coerce", format="mixed"
        )
        invalid = frame["__merge_timestamp"].isna()
        if invalid.any():
            examples = frame.loc[invalid, current_timestamp_name].head(5).tolist()
            raise ValueError(
                f"RV{village} has {int(invalid.sum())} invalid timestamps in {path.name}; "
                f"examples={examples}. No files were deleted."
            )
        input_rows[path.name] = len(frame)
        frames.append(frame)

    assert union_columns and timestamp_name is not None
    for index, frame in enumerate(frames):
        for column in union_columns:
            if column not in frame.columns:
                frame[column] = ""
        frames[index] = frame.loc[:, union_columns + ["__merge_timestamp"]]
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.sort_values("__merge_timestamp", kind="stable").reset_index(drop=True)
    merged = merged.loc[:, union_columns]
    if len(merged) != sum(input_rows.values()):
        raise RuntimeError(f"RV{village} row count changed while combining inputs.")

    output = _merged_output_path(directory, village)
    temporary = directory / f".{output.name}.tmp"
    return {
        "village": village,
        "directory": directory,
        "inputs": inputs,
        "input_rows": input_rows,
        "rows": merged,
        "row_count": len(merged),
        "columns": union_columns,
        "timestamp_column": timestamp_name,
        "output": output,
        "temporary": temporary,
    }


def _write_and_verify(plan: dict) -> None:
    frame = plan["rows"]
    temporary: Path = plan["temporary"]
    frame.to_csv(
        temporary, index=False, sep=";", encoding="utf-8-sig", lineterminator="\n"
    )
    reread = pd.read_csv(
        temporary, sep=";", dtype="string", keep_default_na=False,
        encoding="utf-8-sig", low_memory=False,
    )
    if list(reread.columns) != plan["columns"]:
        raise ValueError(f"RV{plan['village']} output header verification failed.")
    if len(reread) != plan["row_count"]:
        raise ValueError(f"RV{plan['village']} output row count verification failed.")
    if not reread.equals(frame.reset_index(drop=True)):
        raise ValueError(
            f"RV{plan['village']} output values did not round-trip exactly; "
            "source files have not been deleted."
        )


def merge_environment_files() -> list[Path]:
    """Merge configured villages independently; delete inputs only after verification."""
    plans = []
    for village in (1, 2, 3):
        directory_value = settings.ENVIRONMENT_DIRECTORIES.get(village)
        if directory_value is None:
            continue
        directory = Path(directory_value)
        plan = _prepare_merge(directory, village)
        if plan is not None:
            plans.append(plan)

    # Prepare and round-trip verify every village before committing any output.
    try:
        for plan in plans:
            _write_and_verify(plan)
        for plan in plans:
            plan["temporary"].replace(plan["output"])
    except Exception:
        for plan in plans:
            plan["temporary"].unlink(missing_ok=True)
        raise

    outputs = [plan["output"] for plan in plans]
    for plan in plans:
        duplicate_rows = int(plan["rows"].duplicated(keep=False).sum())
        duplicate_timestamps = int(
            plan["rows"][plan["timestamp_column"]].duplicated(keep=False).sum()
        )
        LOGGER.info(
            "RV%s merged %s rows from %s files into %s; exact duplicate rows retained: %s; "
            "repeated timestamps retained: %s",
            plan["village"], plan["row_count"], len(plan["inputs"]),
            plan["output"], duplicate_rows, duplicate_timestamps,
        )

    if settings.DELETE_ENVIRONMENT_INPUTS_AFTER_MERGE:
        # Every merged output is committed and verified before any source is removed.
        for plan in plans:
            for source in plan["inputs"]:
                source.unlink()
            LOGGER.info("Removed %s source files for RV%s after verification.", len(plan["inputs"]), plan["village"])
    else:
        LOGGER.info("Input deletion is disabled in settings.py; all source files were kept.")
    return outputs


def _as_path_list(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(path) for path in value if path is not None]


def _parse_local_times(values: pd.Series, timezone: str) -> pd.Series:
    """Parse naive Academy wall times in the configured local timezone."""
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if parsed.dt.tz is None:
        # A repeated wall time during the autumn clock change cannot be
        # disambiguated from this log format, which contains no UTC offset.
        parsed = parsed.dt.tz_localize(
            ZoneInfo(timezone), ambiguous="NaT", nonexistent="shift_forward"
        )
    else:
        parsed = parsed.dt.tz_convert(ZoneInfo(timezone))
    return parsed.dt.tz_convert("UTC")


def _normalise_subject(value: Any) -> str:
    if pd.isna(value):
        return ""
    subject = str(value).strip().lower()
    aliases = {str(key).lower(): str(val).lower() for key, val in settings.EVENT_SUBJECT_ALIASES.items()}
    return aliases.get(subject, subject)


def _load_environment_data(village: int) -> pd.DataFrame | None:
    """Load the consolidated sensor file for one village, if configured."""
    directory_value = settings.ENVIRONMENT_DIRECTORIES.get(village)
    if directory_value is None:
        return None
    directory = Path(directory_value)
    if not directory.is_dir():
        LOGGER.warning("RV%s environment directory is unavailable: %s", village, directory)
        return None
    paths = sorted(directory.glob(f"environment_rv{village}_*.csv"))
    if not paths:
        LOGGER.warning(
            "No merged environment file found for RV%s in %s; check the configured environment folder.",
            village, directory,
        )
        return None
    if len(paths) != 1:
        raise ValueError(
            f"Expected one merged environment file for RV{village}, found {len(paths)}: "
            f"{[path.name for path in paths]}"
        )

    path = paths[0]
    frame = pd.read_csv(
        path, sep=";", dtype="string", keep_default_na=False,
        encoding="utf-8-sig", low_memory=False,
    )
    if frame.empty:
        raise ValueError(f"Merged RV{village} environment file has no sensor rows: {path}")
    frame.columns = [str(column).strip().lstrip("\ufeff") for column in frame.columns]
    timestamp_columns = [column for column in frame if column.strip().lower() == "timestamp"]
    if len(timestamp_columns) != 1:
        raise ValueError(f"Merged RV{village} environment file must have one timestamp column: {path}")

    # Physical temperature/humidity sensor columns are named with _C or _RH
    # suffixes. control_C is a setpoint, not a sensor reading.
    sensor_columns = [
        column for column in frame.columns
        if re.search(r"_(?:C|RH)$", column, flags=re.IGNORECASE)
        and column.casefold() != "control_c"
    ]
    if not sensor_columns:
        raise ValueError(f"No temperature or humidity sensor columns found in {path}.")

    timestamp_column = timestamp_columns[0]
    frame["__env_time_utc"] = _parse_local_times(
        frame[timestamp_column], settings.ENVIRONMENT_TIMEZONE
    )
    invalid = frame["__env_time_utc"].isna()
    if invalid.any():
        examples = frame.loc[invalid, timestamp_column].head(5).tolist()
        raise ValueError(
            f"Merged RV{village} environment file has {int(invalid.sum())} invalid timestamps; "
            f"examples={examples}"
        )
    for column in sensor_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("__env_time_utc", kind="stable").reset_index(drop=True)


def _attach_environment_to_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """Choose the nearest consolidated sensor row once for each session."""
    result = sessions.copy()
    sensor_data = {village: _load_environment_data(village) for village in (1, 2, 3)}
    sensor_columns_by_village = {
        village: [
            column for column in frame.columns
            if re.search(r"_(?:C|RH)$", column, flags=re.IGNORECASE)
            and column.casefold() != "control_c"
        ] if frame is not None else []
        for village, frame in sensor_data.items()
    }
    environment_output_columns = list(ENVIRONMENT_SESSION_COLUMNS)
    for columns in sensor_columns_by_village.values():
        for column in columns:
            output_column = f"env_{column}"
            if output_column not in environment_output_columns:
                environment_output_columns.append(output_column)

    if result.empty:
        for column in environment_output_columns:
            if column == "env_timestamp":
                result[column] = pd.Series(dtype="datetime64[ns, UTC]")
            elif column == "env_time_from_session_start_seconds" or (
                column.startswith("env_") and column != "env_match_status"
            ):
                result[column] = pd.Series(dtype="float64")
            else:
                result[column] = pd.Series(dtype="string")
        return result

    result["env_timestamp"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    result["env_time_from_session_start_seconds"] = np.nan
    result["env_match_status"] = "no_environment_file"
    for columns in sensor_columns_by_village.values():
        for column in columns:
            result[f"env_{column}"] = np.nan

    if "ratvillage" not in result or "datetime_start_session" not in result:
        return result

    for village, frame in sensor_data.items():
        if frame is None:
            continue
        session_indices = result.index[result["ratvillage"].eq(village)]
        if len(session_indices) == 0:
            continue
        starts = pd.to_datetime(result.loc[session_indices, "datetime_start_session"], utc=True, errors="coerce")
        valid = starts.notna()
        result.loc[session_indices[~valid], "env_match_status"] = "invalid_session_start"
        if not valid.any():
            continue

        valid_indices = session_indices[valid.to_numpy()]
        start_ns = starts.loc[valid_indices].array.asi8
        environment_ns = frame["__env_time_utc"].array.asi8
        right = np.searchsorted(environment_ns, start_ns, side="left")
        left = np.clip(right - 1, 0, len(environment_ns) - 1)
        right_clipped = np.clip(right, 0, len(environment_ns) - 1)
        left_distance = np.abs(start_ns - environment_ns[left])
        right_distance = np.abs(environment_ns[right_clipped] - start_ns)
        # Ties use the earlier reading for deterministic matching.
        choose_right = right_distance < left_distance
        nearest = np.where(choose_right, right_clipped, left)

        result.loc[valid_indices, "env_timestamp"] = pd.to_datetime(
            environment_ns[nearest], utc=True
        )
        result.loc[valid_indices, "env_time_from_session_start_seconds"] = (
            environment_ns[nearest] - start_ns
        ) / 1e9
        result.loc[valid_indices, "env_match_status"] = "matched"
        for column in sensor_columns_by_village[village]:
            result.loc[valid_indices, f"env_{column}"] = frame[column].iloc[nearest].to_numpy()
        LOGGER.info("RV%s environment readings matched to %s sessions.", village, len(valid_indices))
    return result


def _load_events() -> pd.DataFrame:
    pieces = []
    for village in (1, 2, 3):
        for path in _as_path_list(settings.EVENT_FILES.get(village)):
            if not path.is_file():
                raise FileNotFoundError(f"Event file configured for RV{village} not found: {path}")
            frame = pd.read_csv(
                path, sep=";", dtype="string", low_memory=False,
                keep_default_na=True, na_values=["", "NA", "NaN", "nan"],
            )
            required = {"date", "subject", "description", "type"}
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise ValueError(f"Event file {path} is missing columns: {', '.join(missing)}")
            frame = frame[["date", "subject", "description", "type"]].copy()
            frame["ratvillage"] = village
            frame["event_source_file"] = path.name
            pieces.append(frame)
    if not pieces:
        return pd.DataFrame()
    events = pd.concat(pieces, ignore_index=True)
    events["event_time_utc"] = _parse_local_times(events["date"], settings.EVENT_TIMEZONE)
    events["event_type"] = events["type"].astype("string").str.strip().str.upper()
    events["subject_match"] = events["subject"].map(_normalise_subject)
    events["event_description"] = events["description"].astype("string").str.strip()
    if events["event_time_utc"].isna().any():
        bad_rows = int(events["event_time_utc"].isna().sum())
        LOGGER.warning("Ignoring %s event rows with invalid/ambiguous local timestamps.", bad_rows)
        events = events.dropna(subset=["event_time_utc"])
    events = events.drop_duplicates(
        ["ratvillage", "subject_match", "event_time_utc", "event_type", "event_description"]
    )
    return events.sort_values(
        ["ratvillage", "subject_match", "event_time_utc"], kind="stable"
    ).reset_index(drop=True)


def _session_environment(events: pd.DataFrame, village: int,
                         start: pd.Timestamp, end: pd.Timestamp | None,
                         next_start: pd.Timestamp | None) -> tuple[float, float]:
    """Take the first logged temperature and humidity values in a session."""
    rows = events.loc[
        events["ratvillage"].eq(village)
        & events["subject_match"].eq("academy")
        & events["event_type"].eq("ACTION")
        & events["event_time_utc"].ge(start)
    ]
    if end is not None:
        rows = rows.loc[rows["event_time_utc"].le(end)]
    if next_start is not None:
        rows = rows.loc[rows["event_time_utc"].lt(next_start)]
    if rows.empty:
        return float("nan"), float("nan")
    rows = rows.sort_values("event_time_utc", kind="stable")
    temperature = humidity = float("nan")
    for description in rows["event_description"].astype(str):
        if np.isnan(temperature):
            temp = re.search(
                r"temperature\s*[:;=]\s*([-+]?\d+(?:\.\d+)?)", description, re.I
            )
            if temp:
                temperature = float(temp.group(1))
        if np.isnan(humidity):
            moisture = re.search(
                r"humidity\s*[:;=]\s*([-+]?\d+(?:\.\d+)?)", description, re.I
            )
            if moisture:
                humidity = float(moisture.group(1))
        if not np.isnan(temperature) and not np.isnan(humidity):
            break
    return temperature, humidity


def _build_sessions(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=[
            "ratvillage", "subject_match", "event_session_id", "event_session_task",
            "datetime_start_session", "datetime_end_session", "event_session_status",
            "event_temperature", "event_humidity", "_next_session_start",
        ])
    pseudo_subjects = {str(value).lower() for value in settings.EVENT_NON_ANIMAL_SUBJECTS}
    session_events = events.loc[
        events["event_type"].isin(["START", "END"])
        & ~events["subject_match"].isin(pseudo_subjects)
        & events["subject_match"].ne("")
        & events["subject_match"].ne("<NA>")
    ]
    sessions = []
    orphan_ends = 0
    for (village, subject), group in session_events.groupby(["ratvillage", "subject_match"], sort=False):
        opened = []
        counter = 0
        for row in group.sort_values("event_time_utc", kind="stable").itertuples(index=False):
            task = str(row.event_description).strip()
            if row.event_type == "START":
                counter += 1
                item = {
                    "ratvillage": int(village), "subject_match": subject,
                    "event_session_id": counter, "event_session_task": task,
                    "datetime_start_session": row.event_time_utc,
                    "datetime_end_session": pd.NaT,
                    "event_session_status": "missing_end",
                }
                sessions.append(item)
                opened.append(item)
            elif row.event_type == "END":
                preceding = [item for item in opened if item["datetime_start_session"] < row.event_time_utc]
                exact = [item for item in preceding if item["event_session_task"] == task]
                candidates = exact or (preceding if len(preceding) == 1 else [])
                if not candidates:
                    orphan_ends += 1
                    continue
                matched = max(candidates, key=lambda item: item["datetime_start_session"])
                matched["datetime_end_session"] = row.event_time_utc
                matched["event_session_status"] = (
                    "matched_end" if matched["event_session_task"] == task else "end_task_label_mismatch"
                )
                opened.remove(matched)

    result = pd.DataFrame(sessions)
    if result.empty:
        return _build_sessions(pd.DataFrame())
    result = result.sort_values(
        ["ratvillage", "subject_match", "datetime_start_session"], kind="stable"
    ).reset_index(drop=True)
    result["_next_session_start"] = result.groupby(["ratvillage", "subject_match"])[
        "datetime_start_session"
    ].shift(-1)
    # Sensor messages use the shared Academy subject. Stop at the next START
    # in the village, even if it belongs to another subject, to avoid carrying
    # a later session's readings backwards into the current one.
    all_starts = {
        int(village): group["event_time_utc"].sort_values(kind="stable").array.asi8
        for village, group in events.loc[events["event_type"].eq("START")].groupby("ratvillage")
    }
    village_next_starts = []
    for item in result.itertuples(index=False):
        starts = all_starts.get(int(item.ratvillage), np.array([], dtype=np.int64))
        pos = np.searchsorted(starts, item.datetime_start_session.value, side="right")
        village_next_starts.append(pd.NaT if pos >= len(starts) else pd.Timestamp(starts[pos], tz="UTC"))
    result["_next_village_start"] = village_next_starts
    result["event_temperature"] = np.nan
    result["event_humidity"] = np.nan
    for index, item in result.iterrows():
        temp, humidity = _session_environment(
            events, int(item["ratvillage"]),
            item["datetime_start_session"],
            None if pd.isna(item["datetime_end_session"]) else item["datetime_end_session"],
            None if pd.isna(item["_next_village_start"]) else item["_next_village_start"],
        )
        result.at[index, "event_temperature"] = temp
        result.at[index, "event_humidity"] = humidity
    if orphan_ends:
        LOGGER.warning("Ignored %s END events without an unambiguous matching START.", orphan_ends)
    LOGGER.info("Built %s animal session windows from Academy START/END events.", len(result))
    return result


def _attach_sessions(data: pd.DataFrame, sessions: pd.DataFrame, has_events: bool) -> pd.DataFrame:
    if "ratvillage" not in data or "subject" not in data or "datetime" not in data:
        raise ValueError("Session matching requires 'ratvillage', 'subject', and 'datetime' columns.")
    output_columns = [
        column for column in sessions.columns
        if column not in ("ratvillage", "subject_match") and not column.startswith("_")
    ]
    collisions = set(output_columns).intersection(data.columns)
    if collisions:
        raise ValueError(f"Session output columns already exist: {sorted(collisions)}")
    result = data.copy()
    for column in output_columns:
        if column in ("datetime_start_session", "datetime_end_session"):
            result[column] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
        elif column in ("event_temperature", "event_humidity", "env_time_from_session_start_seconds") or (
            column.startswith("env_") and column not in ("env_match_status", "env_timestamp")
        ):
            result[column] = np.nan
        elif column == "env_timestamp":
            result[column] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
        else:
            result[column] = pd.NA
    if not has_events:
        result["event_session_status"] = "no_event_files"
        return result

    if sessions.empty:
        result["event_session_status"] = "no_animal_start_events"
        return result
    time = pd.to_datetime(result["datetime"], errors="coerce", utc=True)
    village = pd.to_numeric(result["ratvillage"].astype("string").str.extract(r"(\d+)$", expand=False), errors="coerce")
    subject = result["subject"].map(_normalise_subject)
    valid_time = time.notna()
    result.loc[~valid_time, "event_session_status"] = "missing_or_invalid_trial_time"

    for (rv, subject_key), trial_index in result.loc[valid_time].groupby(
        [village.loc[valid_time], subject.loc[valid_time]], sort=False
    ).groups.items():
        candidates = sessions.loc[
            sessions["ratvillage"].eq(rv) & sessions["subject_match"].eq(subject_key)
        ].sort_values("datetime_start_session", kind="stable")
        if candidates.empty:
            result.loc[trial_index, "event_session_status"] = "no_session_events"
            continue
        trial_ns = time.loc[trial_index].array.asi8
        start_ns = candidates["datetime_start_session"].array.asi8
        selected = np.searchsorted(start_ns, trial_ns, side="right") - 1
        has_start = selected >= 0
        valid = has_start.copy()
        if has_start.any():
            positions = np.flatnonzero(has_start)
            chosen = selected[positions]
            session_end = candidates["datetime_end_session"].array.asi8[chosen]
            has_end = ~pd.isna(candidates["datetime_end_session"].iloc[chosen]).to_numpy()
            valid[positions[has_end]] &= trial_ns[positions[has_end]] <= session_end[has_end]
            next_start = candidates["_next_session_start"].array.asi8[chosen]
            has_next = next_start != np.iinfo(np.int64).min
            valid[positions[has_next]] &= trial_ns[positions[has_next]] < next_start[has_next]

        chosen_rows = trial_index[valid]
        result.loc[trial_index[~valid], "event_session_status"] = "outside_session_window"
        if len(chosen_rows) == 0:
            continue
        chosen_sessions = candidates.iloc[selected[valid]]
        for column in output_columns:
            result.loc[chosen_rows, column] = chosen_sessions[column].to_numpy()
    return result


def prepare_data(apply_filters: bool = False) -> pd.DataFrame:
    """Merge environment exports, prepare trial rows, and add session data."""
    merge_environment_files()
    data = _prepare_trial_table(apply_filters=apply_filters)
    events = _load_events()
    sessions = _build_sessions(events)
    sessions = _attach_environment_to_sessions(sessions)
    return _attach_sessions(data, sessions, has_events=not events.empty)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = prepare_data()
    output_dir = Path(settings.OUTPUT_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / settings.OUTPUT_FILENAME
    data.to_csv(output_path, sep=settings.CSV_SEPARATOR, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved prepared data with %s rows and %s columns to %s", len(data), len(data.columns), output_path)


if __name__ == "__main__":
    main()

"""Merge and prepare trial data exported by the three Rat Villages."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

import settings

LOGGER = logging.getLogger("reports_nas.data_sorting")


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
    frame["dataset"] = dataset
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
    grouping = [column for column in ("subject", "task", "stage_substage", "session") if column in data.columns]
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
    if {"subject", "session", "datetime", "datetime_end"}.issubset(data.columns):
        grouped = data.dropna(subset=["subject", "session"]).groupby(["subject", "session"], dropna=False)
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


def sort_trial_data() -> pd.DataFrame:
    """Run the first-stage merge, cleaning, and configured task selection."""
    data = load_all_villages()
    if "trial_result" in data.columns:
        data = data.loc[data["trial_result"].notna()].copy()
    data = _add_core_fields(data)
    data = _add_response_fields(data)
    data = _add_session_fields(data)
    data = _add_inactive_hours(data)
    data = apply_task_switches(data)

    if "datetime" in data.columns:
        data = data.sort_values(["ratvillage", "subject", "datetime"], kind="stable", na_position="last")
    return data.reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sorted_data = sort_trial_data()
    output_dir = Path(settings.OUTPUT_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / settings.OUTPUT_FILENAME
    sorted_data.to_csv(output_path, sep=settings.CSV_SEPARATOR, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved %s rows and %s columns to %s", len(sorted_data), len(sorted_data.columns), output_path)


if __name__ == "__main__":
    main()

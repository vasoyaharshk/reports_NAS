"""Safely combine environment exports separately for each Rat Village."""
from __future__ import annotations

import csv
import logging
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import settings

LOGGER = logging.getLogger("reports_nas.environment_merge")
SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}
MERGED_NAME_PATTERN = re.compile(r"^environment_rv[123]_\d{8}_\d{6}\.csv$", re.I)


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
    expected_columns: list[str] | None = None
    input_rows: dict[str, int] = {}

    for path in inputs:
        frame = _read_input(path)
        current_timestamp_name = _timestamp_column(frame, path)
        if expected_columns is None:
            expected_columns = list(frame.columns)
            timestamp_name = current_timestamp_name
        elif set(frame.columns) != set(expected_columns):
            missing = [name for name in expected_columns if name not in frame.columns]
            extra = [name for name in frame.columns if name not in expected_columns]
            raise ValueError(
                f"RV{village} schema mismatch in {path.name}; missing={missing}, extra={extra}. "
                "No files were deleted."
            )
        frame = frame.loc[:, expected_columns]
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

    assert expected_columns is not None and timestamp_name is not None
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.sort_values("__merge_timestamp", kind="stable").reset_index(drop=True)
    merged = merged.loc[:, expected_columns]
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
        "columns": expected_columns,
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    outputs = merge_environment_files()
    for output in outputs:
        LOGGER.info("Environment merge output: %s", output)


if __name__ == "__main__":
    main()

"""Compare RV2/RV3 trial and session counts across configured periods."""
from __future__ import annotations

import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import settings

LOGGER = logging.getLogger("reports_nas.period_comparison")
SUBJECT_COLUMNS = ("subject", "subject_id", "subject_name")
SESSION_CANDIDATES = (
    "session_id", "session_ID", "sessionID", "sessionid", "session_uuid",
    "session_uid", "session_number", "session_no", "session",
    "session_start", "session_start_time", "session_start_datetime",
    "session_start_timestamp",
)


def _periods() -> list[dict]:
    zone = ZoneInfo(settings.PERIOD_COMPARISON_TIMEZONE)
    configured = settings.PERIOD_COMPARISON_PERIODS
    if not configured:
        raise ValueError("PERIOD_COMPARISON_PERIODS is empty in settings.py.")
    parsed = []
    names = set()
    for item in configured:
        name = str(item["period"])
        if name in names:
            raise ValueError(f"Period names must be unique: {name}")
        names.add(name)
        start = pd.Timestamp(item["start"])
        start = start.tz_localize(zone) if start.tzinfo is None else start.tz_convert(zone)
        end_value = item.get("end")
        end = None if end_value is None else pd.Timestamp(end_value)
        if end is not None:
            end = end.tz_localize(zone) if end.tzinfo is None else end.tz_convert(zone)
            if end <= start:
                raise ValueError(f"Period end must be after its start: {name}")
        parsed.append({"period": name, "start": start, "end": end})

    starts = [item["start"] for item in parsed]
    if starts != sorted(starts) or len(starts) != len(set(starts)):
        raise ValueError("Periods must be listed in chronological order with unique start times.")
    if any(item["end"] is None for item in parsed[:-1]):
        raise ValueError("Only the final configured period can have end=None.")
    for previous, current in zip(parsed, parsed[1:]):
        if previous["end"] is not None and current["start"] < previous["end"]:
            raise ValueError("Configured periods overlap.")
    return parsed


def _identify_columns(data: pd.DataFrame) -> tuple[str, str]:
    for required in ("ratvillage", "datetime"):
        if required not in data.columns:
            raise ValueError(f"Sorted data must include a {required!r} column.")
    subject_matches = [name for name in SUBJECT_COLUMNS if name in data.columns]
    if not subject_matches:
        raise ValueError("Could not identify subject column (subject/subject_id/subject_name).")
    configured_session = settings.PERIOD_COMPARISON_SESSION_COLUMN
    if configured_session:
        if configured_session not in data.columns:
            raise ValueError(f"Configured session column not found: {configured_session}")
        return subject_matches[0], configured_session
    for candidate in SESSION_CANDIDATES:
        if candidate in data.columns:
            return subject_matches[0], candidate
    raise ValueError(
        "Could not identify a session ID column. Set PERIOD_COMPARISON_SESSION_COLUMN "
        "in settings.py to the column that uniquely identifies a session."
    )


def _prepare_trials(data: pd.DataFrame, subject_column: str, session_column: str) -> pd.DataFrame:
    # prepare_data.py writes these as UTC-aware timestamps. Parsing with utc=True
    # preserves those instants and gives consistent comparisons with local bounds.
    timestamps = pd.to_datetime(data["datetime"], errors="coerce", utc=True)
    village_number = data["ratvillage"].astype("string").str.extract(r"(\d+)$", expand=False)
    prepared = pd.DataFrame({
        "rv_number": pd.to_numeric(village_number, errors="coerce"),
        "subject": data[subject_column].astype("string").str.strip(),
        "session_key": data[session_column].astype("string").str.strip(),
        "datetime_utc": timestamps,
    }, index=data.index)
    prepared = prepared.loc[
        prepared["rv_number"].isin([2, 3])
        & prepared["subject"].notna()
        & prepared["subject"].ne("")
        & prepared["datetime_utc"].notna()
    ].copy()
    if prepared.empty:
        raise ValueError("No valid RV2/RV3 trial records with subject and datetime were found.")
    local_times = prepared["datetime_utc"].dt.tz_convert(settings.PERIOD_COMPARISON_TIMEZONE)
    prepared["ratvillage"] = "RV" + prepared["rv_number"].astype(int).astype(str)
    prepared["day"] = local_times.dt.date
    return prepared[["ratvillage", "subject", "session_key", "datetime_utc", "day"]]


def _make_period_tables(data: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.Timestamp]:
    periods = _periods()
    subject_column, session_column = _identify_columns(data)
    LOGGER.info("Counting sessions using column: %s", session_column)
    trials = _prepare_trials(data, subject_column, session_column)
    latest_trial = trials["datetime_utc"].max()
    period_pieces = []
    for period in periods:
        start_utc = period["start"].tz_convert("UTC")
        end_utc = period["end"].tz_convert("UTC") if period["end"] is not None else None
        if end_utc is None and start_utc > latest_trial:
            raise ValueError(f"Open-ended period {period['period']!r} starts after the latest trial.")
        mask = trials["datetime_utc"].ge(start_utc)
        if end_utc is not None:
            mask &= trials["datetime_utc"].lt(end_utc)
        selected = trials.loc[mask].copy()
        selected["period"] = period["period"]
        period_pieces.append(selected)
    period_trials = pd.concat(period_pieces, ignore_index=True)
    if period_trials.empty:
        raise ValueError("No trial records fall within the configured periods.")
    invalid_sessions = period_trials["session_key"].isna() | period_trials["session_key"].eq("")
    if invalid_sessions.any():
        raise ValueError(
            f"{int(invalid_sessions.sum())} trial rows have no session ID. "
            f"Check PERIOD_COMPARISON_SESSION_COLUMN (currently {session_column!r})."
        )
    observed_periods = set(period_trials["period"])
    missing = [period["period"] for period in periods if period["period"] not in observed_periods]
    if missing:
        LOGGER.warning("No trial records in configured periods: %s", ", ".join(missing))

    observed_daily = (period_trials.groupby(["period", "ratvillage", "subject", "day"], dropna=False)
                      .agg(trials=("session_key", "size"),
                           sessions=("session_key", "nunique"))
                      .reset_index())
    roster = period_trials[["ratvillage", "subject"]].drop_duplicates()
    zone = ZoneInfo(settings.PERIOD_COMPARISON_TIMEZONE)
    calendar_rows = []
    for period in periods:
        if period["end"] is None:
            last_day = latest_trial.tz_convert(zone).date()
        else:
            last_day = (period["end"] - pd.Timedelta(seconds=1)).date()
        first_day = period["start"].date()
        if last_day < first_day:
            raise ValueError(f"Configured period has no calendar-day coverage: {period['period']}")
        calendar_rows.extend(
            {"period": period["period"], "day": day.date()}
            for day in pd.date_range(first_day, last_day, freq="D")
        )
    calendar = pd.DataFrame(calendar_rows)
    roster["_join"] = 1
    calendar["_join"] = 1
    daily_subject = (roster.merge(calendar, on="_join").drop(columns="_join")
                     .merge(observed_daily, on=["ratvillage", "subject", "period", "day"], how="left")
                     .fillna({"trials": 0, "sessions": 0}))
    daily_subject["trials"] = daily_subject["trials"].astype(int)
    daily_subject["sessions"] = daily_subject["sessions"].astype(int)
    daily_subject["active"] = daily_subject["trials"].gt(0)
    daily_subject = daily_subject.sort_values(["ratvillage", "subject", "period", "day"])

    daily_village = (daily_subject.groupby(["period", "ratvillage", "day"], dropna=False)
                     .agg(total_trials=("trials", "sum"),
                          total_sessions=("sessions", "sum"),
                          active_subjects=("active", "sum"))
                     .reset_index())
    daily_village["trials_per_subject_day"] = daily_village["total_trials"].div(
        daily_village["active_subjects"].where(daily_village["active_subjects"].gt(0))
    ).fillna(0)
    daily_village["sessions_per_subject_day"] = daily_village["total_sessions"].div(
        daily_village["active_subjects"].where(daily_village["active_subjects"].gt(0))
    ).fillna(0)
    village_by_period = (daily_village.groupby(["ratvillage", "period"], dropna=False)
                         .agg(avg_trials_per_subject_per_day=("trials_per_subject_day", "mean"),
                              avg_sessions_per_subject_per_day=("sessions_per_subject_day", "mean"))
                         .reset_index())
    village_by_period.iloc[:, 2:] = village_by_period.iloc[:, 2:].round(2)
    village_by_period = village_by_period.sort_values(["ratvillage", "period"])

    subject_by_period = (daily_subject.groupby(["ratvillage", "subject", "period"], dropna=False)
                         .agg(avg_trials_per_subject_per_day=("trials", "mean"),
                              avg_sessions_per_subject_per_day=("sessions", "mean"))
                         .reset_index())
    subject_by_period.iloc[:, 3:] = subject_by_period.iloc[:, 3:].round(2)
    subject_by_period = subject_by_period.sort_values(["ratvillage", "subject", "period"])

    value_columns = ["avg_trials_per_subject_per_day", "avg_sessions_per_subject_per_day"]
    period_order = [period["period"] for period in periods]
    village_comparison = village_by_period.pivot(index="ratvillage", columns="period", values=value_columns)
    village_comparison.columns = [f"{period} | {metric}" for metric, period in village_comparison.columns]
    village_column_order = [f"{period} | {metric}" for period in period_order for metric in value_columns]
    village_comparison = village_comparison.reset_index()
    village_comparison = village_comparison[["ratvillage", *[c for c in village_column_order if c in village_comparison]]]
    village_comparison.columns.name = None
    subject_comparison = subject_by_period.pivot(index=["ratvillage", "subject"], columns="period", values=value_columns)
    subject_comparison.columns = [f"{period} | {metric}" for metric, period in subject_comparison.columns]
    subject_column_order = [f"{period} | {metric}" for period in period_order for metric in value_columns]
    subject_comparison = subject_comparison.reset_index()
    subject_comparison = subject_comparison[["ratvillage", "subject", *[c for c in subject_column_order if c in subject_comparison]]]
    subject_comparison.columns.name = None

    latest_local = latest_trial.tz_convert(zone)
    period_settings = pd.DataFrame([
        {"period": p["period"],
         "start": p["start"].strftime("%Y-%m-%d %H:%M:%S"),
         "end": p["end"].strftime("%Y-%m-%d %H:%M:%S") if p["end"] is not None else "Latest available record",
         "latest_data_record": latest_local.strftime("%Y-%m-%d %H:%M:%S %Z")}
        for p in periods
    ])
    tables = {
        "Village comparison": village_comparison,
        "Subject comparison": subject_comparison,
        "Village by period": village_by_period,
        "Subject by period": subject_by_period,
        "Period settings": period_settings,
    }
    return tables, latest_local


def generate(data: pd.DataFrame) -> Path:
    """Generate the period-comparison workbook and return its path."""
    tables, latest_local = _make_period_tables(data)
    output_path = Path(settings.REPORT_DIRECTORY) / settings.PERIOD_COMPARISON_OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.book[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                letter = column_cells[0].column_letter
                width = min(max(max(len(str(cell.value or "")) for cell in column_cells) + 2, 12), 48)
                worksheet.column_dimensions[letter].width = width
    LOGGER.info("Period comparison saved to %s", output_path)
    LOGGER.info("Latest trial in dataset: %s", latest_local.strftime("%Y-%m-%d %H:%M:%S %Z"))
    return output_path

"""Attach Academy session events and session-level sensor values to trials."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import settings

LOGGER = logging.getLogger("reports_nas.data_enrichment")
SESSION_OUTPUT_COLUMNS = (
    "event_session_id", "event_session_task", "datetime_start_session",
    "datetime_end_session", "event_session_status", "event_temperature",
    "event_humidity",
)


def _as_path_list(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(path) for path in value if path is not None]


def _parse_local_times(values: pd.Series, timezone: str) -> pd.Series:
    """Parse naive Academy/Excel wall times in the configured local timezone."""
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


def _attach_sessions(data: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if "ratvillage" not in data or "subject" not in data or "datetime" not in data:
        raise ValueError("Session matching requires 'ratvillage', 'subject', and 'datetime' columns.")
    collisions = set(SESSION_OUTPUT_COLUMNS).intersection(data.columns)
    if collisions:
        raise ValueError(f"Session output columns already exist: {sorted(collisions)}")
    result = data.copy()
    for column in SESSION_OUTPUT_COLUMNS:
        if column in ("datetime_start_session", "datetime_end_session"):
            result[column] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
        elif column in ("event_temperature", "event_humidity"):
            result[column] = np.nan
        else:
            result[column] = pd.NA
    if events.empty:
        result["event_session_status"] = "no_event_files"
        return result

    sessions = _build_sessions(events)
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
        for column in SESSION_OUTPUT_COLUMNS:
            result.loc[chosen_rows, column] = chosen_sessions[column].to_numpy()
    return result


def enrich_trial_data(data: pd.DataFrame) -> pd.DataFrame:
    """Attach event-derived session windows and one sensor pair per session.

    Every source trial row and original source column is retained. Event files
    are associated with a village in settings.py; file names do not need an
    ``_rvN`` suffix. Configured environment workbooks are intentionally not
    read by this step.
    """
    events = _load_events()
    return _attach_sessions(data, events)

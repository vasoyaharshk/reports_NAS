"""Weekly trial and occupancy reports, grouped by Rat Village."""
from pathlib import Path
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import settings

LOGGER = logging.getLogger(__name__)
WEEKS_PER_PAGE = 26


def _calendar_and_daily(frame):
    """Build daily village metrics and a denominator for each observed date."""
    trials = frame.groupby(["village", "day"]).size().rename("total_trials").reset_index()
    active = (
        frame.assign(hour=frame["local_datetime"].dt.floor("h"))
        .drop_duplicates(["village", "subject", "day", "hour"])
        .groupby(["village", "day"]).size()
        .rename("active_rat_hours").reset_index()
    )
    daily = trials.merge(active, on=["village", "day"], how="left")

    # The living population is a village/day value, not a per-trial count.
    if "number_of_living_subjects" in frame:
        counts = frame[["village", "day", "number_of_living_subjects"]].copy()
        counts["number_of_living_subjects"] = pd.to_numeric(
            counts["number_of_living_subjects"], errors="coerce"
        )
        inconsistent = counts.groupby(["village", "day"])["number_of_living_subjects"].nunique(dropna=True)
        if inconsistent.gt(1).any():
            raise ValueError("Conflicting living-subject counts within a village/date.")
        population = counts.groupby(["village", "day"])["number_of_living_subjects"].first().rename("living_subjects")
        if population.isna().any() or population.le(0).any() or population.mod(1).ne(0).any():
            raise ValueError("Living-subject counts must be positive integers on every observed date.")
        daily = daily.merge(population.reset_index(), on=["village", "day"], how="left")
        daily["population_source"] = "number_of_living_subjects"
    else:
        LOGGER.warning(
            "number_of_living_subjects is not yet in prepared data; "
            "using provisional first-to-last-observed rat activity spans."
        )
        observed = frame[["village", "subject", "day"]].drop_duplicates()
        spans = observed.groupby(["village", "subject"])["day"].agg(["min", "max"]).reset_index()
        days = []
        for row in spans.itertuples(index=False):
            days.append(pd.DataFrame({
                "village": row.village,
                "day": pd.date_range(row.min, row.max, freq="D"),
                "subject": row.subject,
            }))
        provisional = pd.concat(days, ignore_index=True)
        population = provisional.groupby(["village", "day"]).size().rename("living_subjects")
        daily = daily.merge(population.reset_index(), on=["village", "day"], how="left")
        daily["population_source"] = "provisional_observed_span"

    # Expand dates between the first and last recorded day for each village.
    # With a living-population field, carry its last observed value forward on
    # trial-free dates; population changes on such dates require a daily roster
    # in the next data-sorting update.
    expanded = []
    for village, subset in daily.groupby("village"):
        calendar = pd.DataFrame({"day": pd.date_range(subset["day"].min(), subset["day"].max(), freq="D")})
        calendar["village"] = village
        calendar = calendar.merge(subset, on=["village", "day"], how="left")
        calendar["total_trials"] = calendar["total_trials"].fillna(0).astype(int)
        calendar["active_rat_hours"] = calendar["active_rat_hours"].fillna(0).astype(int)
        if "number_of_living_subjects" in frame:
            calendar["living_subjects"] = calendar["living_subjects"].ffill().bfill()
            calendar["population_source"] = "number_of_living_subjects_carried_on_empty_days"
        else:
            calendar["living_subjects"] = calendar["living_subjects"].fillna(0)
            calendar["population_source"] = "provisional_observed_span"
        expanded.append(calendar)
    return pd.concat(expanded, ignore_index=True)


def _box_minutes(frame):
    """Count each matched session once, not once for every trial in it."""
    required = {"entry_time", "exit_time", "time_spent_in_box"}
    if not required.issubset(frame.columns):
        LOGGER.warning("Session entry/exit columns missing: box occupancy will be unavailable.")
        return pd.DataFrame(columns=["village", "day", "box_minutes"])
    sessions = frame[["village", "subject", "entry_time", "exit_time", "time_spent_in_box"]].copy()
    sessions["entry_time"] = pd.to_datetime(sessions["entry_time"], utc=True, errors="coerce")
    sessions["exit_time"] = pd.to_datetime(sessions["exit_time"], utc=True, errors="coerce")
    sessions["minutes"] = pd.to_numeric(sessions["time_spent_in_box"], errors="coerce")
    sessions = sessions.dropna(subset=["entry_time", "exit_time", "minutes"])
    sessions = sessions.loc[
        sessions["minutes"].ge(0) & sessions["exit_time"].ge(sessions["entry_time"])
    ]
    sessions = sessions.drop_duplicates(["village", "subject", "entry_time", "exit_time"])
    if sessions.empty:
        return pd.DataFrame(columns=["village", "day", "box_minutes"])

    # Split sessions at local midnight so each day's occupancy is accurate.
    records = []
    for row in sessions.itertuples(index=False):
        start = row.entry_time.tz_convert(settings.REPORT_TIMEZONE)
        end = row.exit_time.tz_convert(settings.REPORT_TIMEZONE)
        while start < end:
            next_midnight = start.normalize() + pd.DateOffset(days=1)
            stop = min(end, next_midnight)
            records.append((row.village, start.tz_localize(None).normalize(),
                            (stop - start).total_seconds() / 60))
            start = stop
    if not records:
        return pd.DataFrame(columns=["village", "day", "box_minutes"])
    return (pd.DataFrame(records, columns=["village", "day", "box_minutes"])
            .groupby(["village", "day"], as_index=False)["box_minutes"].sum())


def _draw_pages(pdf, weekly, village, kind):
    for start in range(0, len(weekly), WEEKS_PER_PAGE):
        subset = weekly.iloc[start:start + WEEKS_PER_PAGE]
        x = np.arange(len(subset))
        fig, ax = plt.subplots(figsize=(16, 7.5))
        if kind == "trials":
            lo = subset["min_trials_per_subject_day"].to_numpy(dtype=float)
            hi = subset["max_trials_per_subject_day"].to_numpy(dtype=float)
            mean = subset["mean_trials_per_subject_day"].to_numpy(dtype=float)
            ax.fill_between(x, lo, hi, color="#6495ed", alpha=0.28, label="Minimum–maximum band")
            ax.plot(x, hi, color="#d97706", linestyle="--", marker="^", markersize=3, label="Maximum")
            ax.plot(x, mean, color="#1e3a8a", marker="o", markersize=3, linewidth=2.5, label="Average")
            ax.plot(x, lo, color="#059669", linestyle="--", marker="v", markersize=3, label="Minimum")
            ax.set_ylabel("Trials per rat per day")
            ax.set_title(f"{village}: Weekly daily trials per living rat")
            ax.set_ylim(bottom=0)
        else:
            ax.bar(x - 0.2, subset["trial_hour_occupancy_pct"], width=0.38,
                   color="#253858", label="Trial-hour occupancy")
            ax.bar(x + 0.2, subset["actual_box_occupancy_pct"], width=0.38,
                   color="#9ec5e8", label="Actual time in box")
            ax.set_ylabel("Average occupancy per living rat (% of day)")
            ax.set_title(f"{village}: Weekly trial-hour and actual box occupancy")
            ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(
            subset["week_start"].dt.strftime("%d %b %Y"), rotation=90, fontsize=8
        )
        ax.set_xlabel("Week starting Monday")
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", ncol=4)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def generate(data):
    """Write weekly trial and occupancy graphs into one PDF plus CSV summaries."""
    required = {"ratvillage", "subject", "datetime"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Weekly report missing columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(data["datetime"]) or data["datetime"].dt.tz is None:
        raise ValueError("Weekly report requires timezone-aware parsed datetime.")

    frame = data.dropna(subset=["ratvillage", "subject", "datetime"]).copy()
    if frame.empty:
        return None, None
    numbers = frame["ratvillage"].astype(str).str.extract(r"([0-9]+)$", expand=False)
    if numbers.isna().any():
        raise ValueError("Could not determine Rat Village for every trial.")
    frame["village"] = "RV" + numbers
    frame["local_datetime"] = frame["datetime"].dt.tz_convert(settings.REPORT_TIMEZONE)
    frame["day"] = frame["local_datetime"].dt.tz_localize(None).dt.normalize()

    daily = _calendar_and_daily(frame)
    box = _box_minutes(frame)
    daily = daily.merge(box, on=["village", "day"], how="left")
    daily["box_minutes"] = daily["box_minutes"].fillna(0)
    daily["trial_hour_occupancy_pct"] = (
        daily["active_rat_hours"] / (24 * daily["living_subjects"].replace(0, np.nan)) * 100
    )
    daily["actual_box_occupancy_pct"] = (
        daily["box_minutes"] / (1440 * daily["living_subjects"].replace(0, np.nan)) * 100
    )
    daily["week_start"] = daily["day"] - pd.to_timedelta(daily["day"].dt.weekday, unit="D")

    # Rat-day weighting (Method B). Averages are total activity divided by
    # the sum of living subjects across days, not an unweighted mean of days.
    weekly = (daily.groupby(["village", "week_start"], as_index=False)
              .agg(total_trials=("total_trials", "sum"),
                   active_rat_hours=("active_rat_hours", "sum"),
                   box_minutes=("box_minutes", "sum"),
                   subject_days=("living_subjects", "sum"),
                   observed_days=("day", "size")))
    weekly["mean_trials_per_subject_day"] = weekly["total_trials"] / weekly["subject_days"]
    weekly["trial_hour_occupancy_pct"] = (
        weekly["active_rat_hours"] / (24 * weekly["subject_days"]) * 100
    )
    weekly["actual_box_occupancy_pct"] = (
        weekly["box_minutes"] / (1440 * weekly["subject_days"]) * 100
    )

    # Individual daily extremes, including zero-trial living subjects.
    rat_daily = frame.groupby(["village", "subject", "day"]).size().rename("trials").reset_index()
    extremes = (rat_daily.assign(week_start=lambda d: d["day"] -
                pd.to_timedelta(d["day"].dt.weekday, unit="D"))
                .groupby(["village", "week_start"])["trials"]
                .agg(max_trials_per_subject_day="max", observed_min="min")
                .reset_index())
    weekly = weekly.merge(extremes, on=["village", "week_start"], how="left")
    weekly["min_trials_per_subject_day"] = np.where(
        weekly["subject_days"] > weekly["observed_days"].where(False, 0),
        weekly["observed_min"], weekly["observed_min"]
    )
    # Detect zero-trial rat-days using the daily population and observed rat counts.
    observed_counts = rat_daily.groupby(["village", "day"]).size().rename("observed_rats").reset_index()
    missing_days = daily.merge(observed_counts, on=["village", "day"], how="left")
    missing_days["has_zero"] = missing_days["living_subjects"] > missing_days["observed_rats"].fillna(0)
    zero_weeks = (missing_days.groupby(["village", "week_start"])["has_zero"]
                  .any().rename("has_zero").reset_index())
    weekly = weekly.merge(zero_weeks, on=["village", "week_start"], how="left")
    weekly["min_trials_per_subject_day"] = np.where(
        weekly["has_zero"], 0, weekly["observed_min"]
    )
    weekly = weekly.drop(columns=["observed_min", "has_zero"])
    weekly = weekly.sort_values(["village", "week_start"])

    directory = Path(settings.REPORT_DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "Weekly_Daily_Trials.csv"
    weekly.to_csv(csv_path, index=False, encoding="utf-8-sig")
    daily_path = directory / "Weekly_Daily_Occupancy.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    pdf_path = directory / "Weekly_Daily_Trials.pdf"
    with PdfPages(pdf_path) as pdf:
        for village, subset in weekly.groupby("village", sort=True):
            subset = subset.sort_values("week_start").reset_index(drop=True)
            _draw_pages(pdf, subset, village, "trials")
            _draw_pages(pdf, subset, village, "occupancy")
    return pdf_path, csv_path

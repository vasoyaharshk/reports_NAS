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
EXPERIMENT_FAMILIES = (
    "Working Memory",
    "Early Training",
    "Probability",
    "Cognitive Bias",
)


def _experiment_family(frame):
    """Classify trials into the four experiment families used in the plot."""
    task = frame.get("task", pd.Series("", index=frame.index)).astype("string")
    source = frame.get("dataset", pd.Series("", index=frame.index)).astype("string")
    task_key = task.str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
    source_key = source.str.lower().fillna("")

    # Unmatched tasks are excluded from this experiment-family breakdown.
    family = pd.Series(pd.NA, index=frame.index, dtype="string")
    family.loc[
        task_key.str.contains("workingmemory", na=False)
        | task_key.str.startswith("wm", na=False)
    ] = "Working Memory"
    family.loc[
        task_key.str.contains("cognitivebias", na=False)
        | task_key.str.startswith("cb", na=False)
    ] = "Cognitive Bias"
    family.loc[
        task_key.str.contains("probability|weber", na=False)
        | task_key.str.startswith("pi", na=False)
    ] = "Probability"
    # Source files take precedence for the separate Early Training and WM exports.
    family.loc[
        source_key.str.startswith("early_training")
        | task_key.str.contains("earlytraining", na=False)
    ] = "Early Training"
    family.loc[source_key.str.startswith("working_memory")] = "Working Memory"
    return family


def _calendar_and_daily(frame):
    """Build daily village metrics and a denominator for each observed date."""
    trials = frame.groupby(["village", "day"]).size().rename("total_trials").reset_index()
    active = (
        frame.assign(hour=frame["local_datetime"].dt.tz_convert("UTC").dt.floor("h"))
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


def _draw_year_facets(pdf, weekly, experiment_weekly, village):
    """Draw weekly experiment averages with one vertically stacked panel per year."""
    years = sorted(pd.to_datetime(weekly["week_start"]).dt.year.unique())
    if not years:
        return

    fig, axes = plt.subplots(
        len(years),
        1,
        figsize=(16, max(5.5, 4.2 * len(years))),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    palette = {
        "Working Memory": "#D55E00",
        "Early Training": "#0072B2",
        "Probability": "#CC79A7",
        "Cognitive Bias": "#009E73",
    }

    for ax, year in zip(axes[:, 0], years):
        village_weeks = weekly.loc[
            pd.to_datetime(weekly["week_start"]).dt.year.eq(year)
        ].sort_values("week_start")
        year_experiments = experiment_weekly.loc[
            experiment_weekly["village"].eq(village)
            & pd.to_datetime(experiment_weekly["week_start"]).dt.year.eq(year)
        ]

        for experiment in EXPERIMENT_FAMILIES:
            series = year_experiments.loc[
                year_experiments["experiment"].eq(experiment)
            ].sort_values("week_start")
            if series.empty:
                continue
            ax.plot(
                series["week_start"],
                series["mean_trials_per_living_rat_day"],
                color=palette[experiment],
                linewidth=1.8,
                marker="o",
                markersize=2.5,
                label=experiment,
            )

        ax.set_ylabel("Trials per living rat per day")
        ax.set_ylim(bottom=0)
        ax.set_title(str(year), loc="left", fontsize=11, weight="bold")
        # Show every observed week, rather than only one tick every four weeks.
        ticks = village_weeks["week_start"]
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            pd.to_datetime(ticks).dt.strftime("%d %b"),
            rotation=90,
            fontsize=7,
        )
        ax.set_xlabel("Week starting Monday")
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color=palette[family], linewidth=2, label=family)
        for family in EXPERIMENT_FAMILIES
    ]
    fig.suptitle(
        f"{village}: weekly trial averages by experiment",
        fontsize=15,
        weight="bold",
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig, bbox_inches="tight")
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
    # Missing session matches are unknown, not proof of zero box time.
    # Trial-free days can be treated as zero, but days with trials and no
    # matched sessions must remain missing.
    daily.loc[daily["total_trials"].eq(0), "box_minutes"] = 0
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

    # Add an experiment-family breakdown on the same living-rat-day denominator.
    # Keep each family within its observed run span: dates before it began or
    # after it ended are not zero-trial experiment days.
    frame["experiment"] = _experiment_family(frame)
    category_daily = (
        frame.loc[frame["experiment"].isin(EXPERIMENT_FAMILIES)]
        .groupby(["village", "experiment", "day"])
        .size().rename("total_trials").reset_index()
    )
    experiment_weekly_parts = []
    for village, village_days in daily.groupby("village", sort=True):
        categories = [
            family for family in EXPERIMENT_FAMILIES
            if family in set(category_daily.loc[
                category_daily["village"].eq(village), "experiment"
            ])
        ]
        if not categories:
            continue
        counts = category_daily.loc[category_daily["village"].eq(village)]
        experiment_parts = []
        for experiment in categories:
            experiment_counts = counts.loc[counts["experiment"].eq(experiment)]
            active_days = village_days.loc[
                village_days["day"].between(
                    experiment_counts["day"].min(), experiment_counts["day"].max()
                ),
                ["day", "living_subjects"],
            ].copy()
            active_days["village"] = village
            active_days["experiment"] = experiment
            active_days = active_days.merge(
                experiment_counts,
                on=["village", "experiment", "day"],
                how="left",
                validate="one_to_one",
            )
            active_days["total_trials"] = active_days["total_trials"].fillna(0).astype(int)
            active_days["week_start"] = active_days["day"] - pd.to_timedelta(
                active_days["day"].dt.weekday, unit="D"
            )
            experiment_parts.append(
                active_days.groupby(
                    ["village", "experiment", "week_start"], as_index=False
                ).agg(
                    total_trials=("total_trials", "sum"),
                    living_subject_days=("living_subjects", "sum"),
                )
            )
        part = pd.concat(experiment_parts, ignore_index=True)
        part["mean_trials_per_living_rat_day"] = (
            part["total_trials"] / part["living_subject_days"]
        )
        experiment_weekly_parts.append(part)
    experiment_weekly = pd.concat(experiment_weekly_parts, ignore_index=True)
    experiment_weekly = experiment_weekly.sort_values(
        ["village", "experiment", "week_start"]
    )
    weekly["trial_hour_occupancy_pct"] = (
        weekly["active_rat_hours"] / (24 * weekly["subject_days"]) * 100
    )
    weekly["actual_box_occupancy_pct"] = (
        weekly["box_minutes"] / (1440 * weekly["subject_days"]) * 100
    )
    # A missing session match cannot be interpreted as zero occupancy.
    missing_box = (daily.assign(missing_box=daily["box_minutes"].isna())
                   .groupby(["village", "week_start"])["missing_box"].any().reset_index())
    weekly = weekly.merge(missing_box, on=["village", "week_start"], how="left")
    weekly.loc[weekly["missing_box"], "actual_box_occupancy_pct"] = np.nan
    weekly = weekly.drop(columns="missing_box")

    # Individual daily extremes, including zero-trial living subjects.
    rat_daily = frame.groupby(["village", "subject", "day"]).size().rename("trials").reset_index()
    extremes = (rat_daily.assign(week_start=lambda d: d["day"] -
                pd.to_timedelta(d["day"].dt.weekday, unit="D"))
                .groupby(["village", "week_start"])["trials"]
                .agg(max_trials_per_subject_day="max", observed_min="min")
                .reset_index())
    weekly = weekly.merge(extremes, on=["village", "week_start"], how="left")
    # Detect zero-trial rat-days using the daily population and observed rat counts.
    observed_counts = rat_daily.groupby(["village", "day"]).size().rename("observed_rats").reset_index()
    missing_days = daily.merge(observed_counts, on=["village", "day"], how="left")
    missing_days["has_zero"] = missing_days["living_subjects"] > missing_days["observed_rats"].fillna(0)
    zero_weeks = (missing_days.groupby(["village", "week_start"])["has_zero"]
                  .any().rename("has_zero").reset_index())
    weekly = weekly.merge(zero_weeks, on=["village", "week_start"], how="left")
    weekly["min_trials_per_subject_day"] = np.where(
        weekly["has_zero"], 0, weekly["observed_min"].fillna(0)
    )
    weekly = weekly.drop(columns=["observed_min", "has_zero"])
    weekly = weekly.sort_values(["village", "week_start"])

    directory = Path(settings.REPORT_DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "Weekly_Daily_Trials.csv"
    weekly.to_csv(csv_path, index=False, encoding="utf-8-sig")
    experiment_csv_path = directory / "Weekly_Daily_Trials_By_Experiment.csv"
    experiment_weekly.to_csv(experiment_csv_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved weekly trial averages by experiment to %s", experiment_csv_path)
    daily_path = directory / "Weekly_Daily_Occupancy.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    pdf_path = directory / "Weekly_Daily_Trials.pdf"
    with PdfPages(pdf_path) as pdf:
        for village, subset in weekly.groupby("village", sort=True):
            subset = subset.sort_values("week_start").reset_index(drop=True)
            _draw_year_facets(pdf, subset, experiment_weekly, village)
    return pdf_path, csv_path


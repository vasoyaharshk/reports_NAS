"""Weekly trial and occupancy reports, grouped by Rat Village."""
from pathlib import Path
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch, Rectangle
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

# Curated diary changes that are relevant to interpreting weekly trial rates.
# Keep these labels short enough for the chart callouts while preserving the
# practical change and its date.
INTERVENTION_EVENTS = {
    "RV1": [
        ("2024-04-30", "Sessions: inter-session interval 5 to 3 h"),
        ("2024-06-04", "Reward: sucrose concentration 20% to 15%"),
        ("2024-06-18", "CA water 3% to 4%; interval changed to 4 h"),
        ("2024-06-27", "Reward volume 25 to 50 µL"),
        ("2024-06-29", "Gal and Sparky 50 µL; other rats 25 µL"),
        ("2024-07-30", "Ciri and Sorrel assigned 40 µL"),
        ("2024-08-21", "CA water 4% to 5%"),
        ("2024-09-04", "Ciri, Sorrel and Luna reduced to 25 µL"),
        ("2024-09-17", "Stimulus mask removed"),
        ("2024-09-27", "Ciri, Joy, Luna and Sorrel reward 25 to 50 µL"),
    ],
    "RV2": [
        ("2025-06-23", "All rats: 2 Automatic Water sessions, 300 µL/trial"),
        ("2025-06-25", "Reward volume increased to 140 µL"),
        ("2025-07-04", "TouchTeaching advancement criterion reduced to 50 trials"),
        ("2025-07-22", "CA water 3% to 3.5%"),
        ("2025-07-28", "Side walls installed; speakers repositioned"),
        ("2025-08-18", "Daily food allocation set to 160 g"),
        ("2025-08-27", "Sucrose concentration 10% to 15%"),
        ("2025-08-28", "Daily food allocation 160 to 180 g"),
        ("2025-09-05", "Soundproofing completed"),
        ("2025-09-22", "Door 3 became functional"),
        ("2025-09-26", "Reward changed to plain water"),
        ("2025-10-14", "Plain-water reward 140 to 70 µL"),
        ("2025-10-17", "Automatic Water disabled"),
        ("2025-10-30", "Plain-water reward 70 to 40 µL"),
        ("2025-10-31", "Plain-water reward 40 to 280 µL"),
        ("2025-11-14", "Reward changed to 220 µL of 5% sucrose"),
        ("2026-04-30", "Cognitive Bias: outside touches became retryable"),
        ("2026-10-19", "All rats on TouchTeaching; reward to 20 µL of 5% sucrose"),
        ("2026-10-22", "Sucrose increased from 5% to 10%; volume stayed 20 µL"),
        ("2026-10-26", "Reward volume increased from 20 to 75 µL"),
        ("2026-10-30", "Reward volume increased from 75 to 100 µL"),
    ],
    "RV3": [
        ("2026-10-19", "All rats on TouchTeaching; reward set to 20 µL; CA water 1% to 2%"),
        ("2026-10-22", "Sucrose increased from 5% to 10%"),
        ("2026-10-26", "Reward volume increased from 20 to 75 µL"),
        ("2026-10-30", "Reward volume reduced from 75 to 70 µL"),
    ],
}

# Concentration windows are local Spain time and end-exclusive. Change dates
# without a recorded time are excluded on the transition day to avoid mixing
# two reward or CA conditions in one daily rate.
CONCENTRATION_WINDOWS = [
    ("RV1", "sucrose", "20%", None, "2024-06-04 00:00"),
    ("RV1", "sucrose", "15%", "2024-06-05 00:00", None),
    ("RV2", "sucrose", "10%", None, "2025-08-27 00:00"),
    ("RV2", "sucrose", "15%", "2025-08-28 00:00", "2025-09-26 00:00"),
    ("RV2", "sucrose", "10%", "2026-10-22 16:00", None),
    # This is the clearly dated 2026 5% period. The 14-Nov-2025 5% entry has
    # no recorded stop date before the diary later reports plain water.
    ("RV2", "sucrose", "5%", "2026-10-20 00:00", "2026-10-22 16:00"),
    ("RV3", "sucrose", "5%", None, "2026-10-22 16:00"),
    ("RV3", "sucrose", "10%", "2026-10-22 16:00", None),
    ("RV1", "CA water", "3%", None, "2024-06-18 16:00"),
    ("RV1", "CA water", "4%", "2024-06-18 16:00", "2024-08-21 17:00"),
    ("RV1", "CA water", "5%", "2024-08-21 17:00", None),
    ("RV2", "CA water", "3%", None, "2025-07-22 12:00"),
    ("RV2", "CA water", "3.5%", "2025-07-22 12:00", "2025-07-25 12:00"),
    # The workbook records 4% and then 5% at the same time; there is no
    # recorded 4% interval, so the next analyzable condition is 5%.
    ("RV2", "CA water", "5%", "2025-07-25 12:00", None),
    ("RV3", "CA water", "1%", None, "2026-10-19 00:00"),
    ("RV3", "CA water", "2%", "2026-10-19 00:00", None),
]


def _experiment_family(frame):
    """Classify trials into the four experiment families used in the plot."""
    task = frame.get("task", pd.Series("", index=frame.index)).astype("string")
    task_key = task.str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)

    # Unmatched tasks are excluded from this experiment-family breakdown.
    family = pd.Series(pd.NA, index=frame.index, dtype="string")
    family.loc[
        task_key.str.contains("workingmemory", na=False)
        | task_key.str.startswith("wm", na=False)
        | task_key.str.startswith("stagetraining", na=False)
    ] = "Working Memory"
    family.loc[
        task_key.str.contains("cognitivebias", na=False)
        | task_key.str.startswith("cb", na=False)
    ] = "Cognitive Bias"
    family.loc[
        task_key.str.contains("probability|weber", na=False)
        | task_key.str.startswith("pi", na=False)
    ] = "Probability"
    # Early Training is defined by these task names, independent of source file.
    early_training_task = task_key.str.contains(
        "touchteaching|lickteaching|habituation",
        na=False,
    )
    family.loc[early_training_task] = "Early Training"
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



def _season_handles():
    """Northern Hemisphere meteorological seasons."""
    return [
        Patch(facecolor="#E5E7EB", label="Winter (Dec–Feb)"),
        Patch(facecolor="#DFF0DB", label="Spring (Mar–May)"),
        Patch(facecolor="#FFFFFF", edgecolor="#D1D5DB", label="Summer (Jun–Aug)"),
        Patch(facecolor="#FCE5D0", label="Autumn (Sep–Nov)"),
    ]


def _shade_seasons(ax, year):
    """Shade exact seasonal boundaries without expanding the x-axis range."""
    spans = [
        (f"{year}-01-01", f"{year}-03-01", "#E5E7EB"),
        (f"{year}-03-01", f"{year}-06-01", "#DFF0DB"),
        (f"{year}-06-01", f"{year}-09-01", "#FFFFFF"),
        (f"{year}-09-01", f"{year}-12-01", "#FCE5D0"),
        (f"{year}-12-01", f"{year + 1}-01-01", "#E5E7EB"),
    ]
    xlimits = ax.get_xlim()
    for start, end, color in spans:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   facecolor=color, alpha=0.85, zorder=-5, linewidth=0)
    ax.set_xlim(xlimits)


def _living_roster(frame):
    """Rat-day roster from health data; both start and death dates inclusive."""
    workbook = Path(settings.RAT_HEALTH_FILE)
    if not workbook.is_file():
        raise FileNotFoundError(f"Weekly report needs rat health workbook: {workbook}")
    details = pd.read_excel(workbook, sheet_name=settings.RAT_DETAILS_SHEET)
    required = {"ratvillage_number", "rat_id", "exp_start_date", "passed_away_date"}
    if not required.issubset(details.columns):
        raise ValueError(f"rat_details missing columns: {sorted(required - set(details.columns))}")
    details = details.copy()
    details["subject"] = details["rat_id"].astype("string").str.strip().str.lower()
    details["ratvillage_number"] = pd.to_numeric(details["ratvillage_number"], errors="coerce")
    details = details.loc[
        details["ratvillage_number"].isin([1, 2, 3])
        & details["subject"].notna() & details["subject"].ne("")
    ].copy()
    if details.empty or details.duplicated(["ratvillage_number", "subject"]).any():
        raise ValueError("Empty or duplicate rat_details subjects")
    details["village"] = "RV" + details["ratvillage_number"].astype(int).astype(str)
    for field in ("exp_start_date", "passed_away_date"):
        details[field] = pd.to_datetime(details[field], errors="coerce").dt.normalize()
    if details["exp_start_date"].isna().any():
        raise ValueError("Invalid experiment start date in rat_details")
    if (details["passed_away_date"].notna()
        & details["passed_away_date"].lt(details["exp_start_date"])).any():
        raise ValueError("Death date precedes experiment start in rat_details")

    parts = []
    for village, trials in frame.groupby("village"):
        members = details.loc[details["village"].eq(village),
                              ["village", "subject", "exp_start_date", "passed_away_date"]]
        if members.empty:
            raise ValueError(f"No rat health roster for {village}")
        calendar = pd.DataFrame({"day": pd.date_range(
            trials["day"].min(), trials["day"].max(), freq="D")})
        calendar["village"] = village
        candidates = calendar.merge(members, on="village")
        alive = candidates.loc[
            candidates["day"].ge(candidates["exp_start_date"])
            & (candidates["passed_away_date"].isna()
               | candidates["day"].le(candidates["passed_away_date"])),
            ["village", "subject", "day"]
        ]
        parts.append(alive)
    roster = pd.concat(parts, ignore_index=True)
    if roster.empty or roster.duplicated(["village", "subject", "day"]).any():
        raise ValueError("Empty or duplicate living-rat roster")
    roster["week_start"] = roster["day"] - pd.to_timedelta(
        roster["day"].dt.weekday, unit="D")
    return roster


def _subject_rates(eligible, trials, period_keys):
    """Shared Method 1 for both graphs, including explicitly zero-trial rats."""
    keys = [*period_keys, "subject"]
    if eligible.duplicated(keys).any():
        raise ValueError("Duplicate eligible rat-period records")
    observed = trials.groupby(keys, as_index=False).agg(
        total_trials=("day", "size"), unique_trial_days=("day", "nunique"))
    rates = eligible.merge(observed, on=keys, how="left", validate="one_to_one")
    rates["total_trials"] = rates["total_trials"].fillna(0).astype(int)
    rates["unique_trial_days"] = rates["unique_trial_days"].fillna(0).astype(int)
    rates["subject_daily_average"] = (
        rates["total_trials"]
        .div(rates["unique_trial_days"].replace(0, np.nan))
        .fillna(0.0)
    )
    return rates


def _summarise_rates(rates, period_keys):
    return (
        rates.assign(has_trials=rates["total_trials"].gt(0))
        .groupby(period_keys, as_index=False)
        .agg(total_trials=("total_trials", "sum"),
             subjects_with_trials=("has_trials", "sum"),
             eligible_subjects=("subject", "nunique"),
             mean_trials_per_subject_day=("subject_daily_average", "mean"))
    )


def _boxplot_subject_rates(frame):
    """Return one Method 1 daily rate per rat for each requested measure."""
    records = []

    def add_trial_rate(source, metric, category_column):
        if source.empty:
            return
        keys = ["village", "subject", category_column, "day"]
        daily_counts = source.groupby(keys, as_index=False).size()
        rates = (daily_counts.groupby(
            ["village", "subject", category_column], as_index=False
        ).agg(total_trials=("size", "sum"), trial_days=("day", "nunique")))
        rates["value"] = rates["total_trials"] / rates["trial_days"]
        rates["metric"] = metric
        rates = rates.rename(columns={category_column: "category"})
        records.append(rates[["village", "subject", "metric", "category", "value"]])

    all_trials = frame.assign(plot_category="All trials")
    add_trial_rate(all_trials, "Trials per rat per trial day", "plot_category")

    experiment_frame = frame.copy()
    experiment_frame["experiment"] = _experiment_family(experiment_frame)
    experiment_frame = experiment_frame.loc[experiment_frame["experiment"].notna()].copy()
    experiment_labels = {
        "Working Memory": "Working Memory",
        "Probability": "Probabilistic Inference",
        "Cognitive Bias": "Cognitive Bias",
    }
    experiment_frame = experiment_frame.loc[
        experiment_frame["experiment"].isin(experiment_labels)
    ].copy()
    experiment_frame["plot_category"] = experiment_frame["experiment"].map(experiment_labels)
    add_trial_rate(experiment_frame, "Trials per rat per trial day by experiment", "plot_category")

    month = frame["day"].dt.month
    seasonal = frame.assign(
        plot_category=np.select(
            [month.isin([12, 1, 2]), month.isin([3, 4, 5]),
             month.isin([6, 7, 8]), month.isin([9, 10, 11])],
            ["Winter", "Spring", "Summer", "Autumn"],
            default="Unknown",
        )
    )
    add_trial_rate(seasonal, "Trials per rat per trial day by season", "plot_category")

    # Session IDs are present in the trial exports and identify sessions
    # across task rows. Count each rat/session once, then divide by the rat's
    # distinct trial dates, using the same Method 1 denominator.
    observed_days = frame.groupby(["village", "subject"])["day"].nunique()
    if "session" in frame.columns:
        sessions = frame[["village", "subject", "session"]].copy()
        sessions["session_key"] = sessions["session"].astype("string").str.strip()
        sessions = sessions.dropna(subset=["session_key"])
        sessions = sessions.loc[sessions["session_key"].ne("")]
        sessions = sessions.drop_duplicates(["village", "subject", "session_key"])
        session_rates = sessions.groupby(["village", "subject"]).size().rename(
            "total_sessions"
        ).to_frame().join(observed_days.rename("trial_days"), how="inner").reset_index()
        session_rates["value"] = session_rates["total_sessions"] / session_rates["trial_days"]
        session_rates["metric"] = "Sessions per rat per trial day"
        session_rates["category"] = "Sessions"
        records.append(session_rates[["village", "subject", "metric", "category", "value"]])
    else:
        LOGGER.warning("Session IDs are missing; the sessions box plot will show no data.")

    # Calculate condition-specific Method 1 rates from documented intervention
    # windows. Several separate periods at the same concentration are combined
    # before one value per rat is calculated.
    condition_parts = {}
    for village, kind, level, start, end in CONCENTRATION_WINDOWS:
        subset = frame.loc[frame["village"].eq(village)].copy()
        mask = pd.Series(True, index=subset.index)
        if start is not None:
            lower = pd.Timestamp(start).tz_localize(settings.REPORT_TIMEZONE)
            mask &= subset["local_datetime"].ge(lower)
        if end is not None:
            upper = pd.Timestamp(end).tz_localize(settings.REPORT_TIMEZONE)
            mask &= subset["local_datetime"].lt(upper)
        subset = subset.loc[mask].copy()
        condition_parts.setdefault((village, kind, level), []).append(subset)

    condition_labels = {
        "sucrose": ("Trials per rat per trial day by sucrose concentration", "sucrose"),
        "CA water": ("Trials per rat per trial day by CA water concentration", "CA water"),
    }
    for (_, kind, level), parts in condition_parts.items():
        subset = pd.concat(parts, ignore_index=False)
        if subset.empty:
            continue
        metric, _ = condition_labels[kind]
        subset["plot_category"] = level
        add_trial_rate(subset, metric, "plot_category")

    if not records:
        return pd.DataFrame(columns=["village", "subject", "metric", "category", "value"])
    return pd.concat(records, ignore_index=True).dropna(subset=["value"])


def _boxplot_summary(subject_rates):
    """Summarise the subject-level distributions displayed by the box plots."""
    if subject_rates.empty:
        return pd.DataFrame(columns=[
            "village", "metric", "category", "subjects", "mean", "minimum", "maximum",
            "q1", "median", "q3",
        ])
    return (subject_rates.groupby(["village", "metric", "category"], as_index=False)
            .agg(subjects=("subject", "nunique"), mean=("value", "mean"),
                 minimum=("value", "min"), maximum=("value", "max"),
                 q1=("value", lambda values: values.quantile(0.25)),
                 median=("value", "median"),
                 q3=("value", lambda values: values.quantile(0.75))))


def _draw_boxplot_panel(ax, subject_rates, metric, categories, village_order):
    """Draw grouped box plots with mean markers and min/max whiskers."""
    palette = {"RV1": "#0072B2", "RV2": "#D55E00", "RV3": "#009E73"}
    positions = []
    values = []
    colors = []
    labels = []
    missing_positions = []
    box_width = 0.62 if len(categories) == 1 else 0.62 / max(len(village_order), 1)
    offsets = np.linspace(-0.32, 0.32, len(village_order)) if village_order else []

    for category_index, category in enumerate(categories):
        for village_index, village in enumerate(village_order):
            subset = subject_rates.loc[
                subject_rates["metric"].eq(metric)
                & subject_rates["category"].eq(category)
                & subject_rates["village"].eq(village)
            ]["value"].dropna().astype(float)
            if len(categories) == 1:
                position = village_index + 1
            else:
                position = category_index + 1 + offsets[village_index]
            if subset.empty:
                missing_positions.append(position)
                continue
            positions.append(position)
            values.append(subset.to_numpy())
            colors.append(palette.get(village, "#6B7280"))
            labels.append((position, village, subset))

    if values:
        artists = ax.boxplot(
            values,
            positions=positions,
            widths=box_width * 0.68,
            patch_artist=True,
            showmeans=True,
            meanline=False,
            whis=(0, 100),
            showfliers=False,
            medianprops={"color": "#111827", "linewidth": 1.2},
            meanprops={"marker": "D", "markerfacecolor": "white",
                       "markeredgecolor": "#111827", "markersize": 5},
            whiskerprops={"color": "#374151", "linewidth": 1.0},
            capprops={"color": "#374151", "linewidth": 1.0},
        )
        for box, color in zip(artists["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.62)

        for position, _, subject_values in labels:
            # Subject points keep small groups and collapsed boxes interpretable.
            offsets_for_points = ((np.arange(len(subject_values)) % 5) - 2) * box_width * 0.12
            ax.scatter(position + offsets_for_points, subject_values,
                       s=13, color="#111827", alpha=0.58, zorder=4)

    if len(categories) == 1:
        centers = np.arange(1, len(village_order) + 1)
        tick_labels = village_order
    else:
        centers = np.arange(1, len(categories) + 1)
        tick_labels = categories
    ax.set_xticks(centers)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=8)
    ax.set_xlim(0.5, len(centers) + 0.5)
    ax.set_title(metric, fontsize=11, weight="bold")
    ax.set_ylabel("Per-rat average per observed trial day")
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    ymax = ax.get_ylim()[1]
    for position in missing_positions:
        ax.text(position, ymax * 0.035, "No data", rotation=90,
                ha="center", va="bottom", fontsize=7, color="#6B7280")
    return palette


def _draw_boxplot_pages(pdf, subject_rates):
    """Add overall, task-family, and seasonal distributions to the PDF."""
    village_order = ["RV1", "RV2", "RV3"]
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    palette = {"RV1": "#0072B2", "RV2": "#D55E00", "RV3": "#009E73"}
    legend = [Patch(facecolor=palette[v], alpha=0.62, label=v) for v in village_order]
    legend.append(Line2D([0], [0], marker="D", linestyle="None",
                         markerfacecolor="white", markeredgecolor="#111827",
                         label="Mean"))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), squeeze=False)
    _draw_boxplot_panel(axes[0, 0], subject_rates,
                        "Trials per rat per trial day", ["All trials"], village_order)
    _draw_boxplot_panel(axes[1, 0], subject_rates,
                        "Sessions per rat per trial day", ["Sessions"], village_order)
    for ax in axes[:, 0]:
        ax.legend(handles=legend, loc="upper right", frameon=False, ncol=3)
    fig.suptitle("Full-period per-rat distributions: overall measures",
                 fontsize=14, weight="bold")
    fig.text(0.01, 0.01,
             "Boxes show the interquartile range, the diamond marks the mean, "
             "and whiskers span the minimum to maximum. Each point is one rat.",
             fontsize=8, color="#4B5563")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    experiment_categories = ["Working Memory", "Probabilistic Inference", "Cognitive Bias"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), squeeze=False)
    for ax, category in zip(axes[0], experiment_categories):
        _draw_boxplot_panel(ax, subject_rates,
                            "Trials per rat per trial day by experiment",
                            [category], village_order)
        ax.set_title(category, fontsize=11, weight="bold")
        ax.legend(handles=legend, loc="upper right", frameon=False, ncol=1)
    fig.suptitle("Full-period per-rat average trials per day by experiment",
                 fontsize=14, weight="bold")
    fig.text(0.01, 0.01,
             "Boxes show the interquartile range, the diamond marks the mean, "
             "and whiskers span the minimum to maximum. Each point is one rat.",
             fontsize=8, color="#4B5563")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    sucrose_levels = ["20%", "15%", "10%", "5%"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True, squeeze=False)
    for ax, village in zip(axes[0], village_order):
        _draw_boxplot_panel(
            ax, subject_rates,
            "Trials per rat per trial day by sucrose concentration",
            sucrose_levels, [village],
        )
        ax.set_ylim(0, 150)
        ax.set_title(village, fontsize=11, weight="bold")
        ax.set_ylabel("Per-rat average per observed trial day" if village == village_order[0] else "")
    fig.suptitle("Average trials per day by reward sucrose concentration across Rat Villages",
                 fontsize=14, weight="bold")
    fig.text(0.01, 0.01,
             "Documented concentration windows only. Boxes show the interquartile range; "
             "diamonds mark means and whiskers span minimum to maximum.",
             fontsize=8, color="#4B5563")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    ca_levels = ["3%", "5%", "1%"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True, squeeze=False)
    for ax, village in zip(axes[0], village_order):
        _draw_boxplot_panel(
            ax, subject_rates,
            "Trials per rat per trial day by CA water concentration",
            ca_levels, [village],
        )
        ax.set_ylim(0, 150)
        ax.set_title(village, fontsize=11, weight="bold")
        ax.set_ylabel("Per-rat average per observed trial day" if village == village_order[0] else "")
    fig.suptitle("Average trials per day by CA water concentration across Rat Villages",
                 fontsize=14, weight="bold")
    fig.text(0.01, 0.01,
             "Documented concentration windows only. Boxes show the interquartile range; "
             "diamonds mark means and whiskers span minimum to maximum.",
             fontsize=8, color="#4B5563")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    seasons = ["Summer", "Winter", "Autumn", "Spring"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True, squeeze=False)
    for ax, season in zip(axes.flat, seasons):
        _draw_boxplot_panel(ax, subject_rates,
                            "Trials per rat per trial day by season",
                            [season], village_order)
        ax.set_ylim(0, 150)
        ax.set_title(season, fontsize=11, weight="bold")
        ax.legend(handles=legend, loc="upper right", frameon=False, ncol=3)
    fig.suptitle("Full-period per-rat average trials per day by season",
                 fontsize=14, weight="bold")
    fig.text(0.01, 0.01,
             "Boxes show the interquartile range, the diamond marks the mean, "
             "and whiskers span the minimum to maximum. Each point is one rat.",
             fontsize=8, color="#4B5563")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _comparison_stat(subject_rates, village, metric, category):
    subset = subject_rates.loc[
        subject_rates["village"].eq(village)
        & subject_rates["metric"].eq(metric)
        & subject_rates["category"].eq(category), "value"
    ].dropna()
    if subset.empty:
        return "No data"
    return (
        f"{subset.mean():.1f} [{subset.min():.1f}–{subset.max():.1f}], "
        f"n={subset.size}"
    )


def _draw_comparison_summary_pages(pdf, details, subject_rates):
    """Add village characteristics and mean/min/max summary tables to the PDF."""
    village_order = ["RV1", "RV2", "RV3"]
    roster = details.copy()
    roster["ratvillage_number"] = pd.to_numeric(roster["ratvillage_number"], errors="coerce")
    roster["subject"] = roster["rat_id"].astype("string").str.strip()
    roster = roster.loc[
        roster["ratvillage_number"].isin([1, 2, 3])
        & roster["subject"].notna() & roster["subject"].ne("")
    ].copy()
    roster["village"] = "RV" + roster["ratvillage_number"].astype(int).astype(str)
    roster["exp_start_date"] = pd.to_datetime(roster["exp_start_date"], errors="coerce")
    roster["age_start_of_exp_weeks"] = pd.to_numeric(
        roster["age_start_of_exp_weeks"], errors="coerce")

    def cohort_description(village):
        members = roster.loc[roster["village"].eq(village)]
        if members.empty:
            return "No rat details"
        starts = members.groupby("exp_start_date").size().sort_index()
        if village == "RV1" and len(starts) > 1:
            return f"{int(starts.iloc[0])} initially; {len(members)} total"
        return str(len(members))

    def age_description(village):
        ages = roster.loc[roster["village"].eq(village), "age_start_of_exp_weeks"].dropna()
        if ages.empty:
            return "No age data"
        low, high = ages.min(), ages.max()
        return f"{low:g} weeks" if low == high else f"{low:g}–{high:g} weeks"

    def gender_description(village):
        genders = (roster.loc[roster["village"].eq(village), "gender"]
                   .dropna().astype(str).str.strip().str.lower().unique())
        return ", ".join(sorted(genders)) if len(genders) else "No data"

    context = {
        "Number of subjects": {
            v: cohort_description(v) for v in village_order
        },
        "Gender": {
            v: gender_description(v) for v in village_order
        },
        "Age at experiment start": {
            v: age_description(v) for v in village_order
        },
        "CA water progression": {
            "RV1": "3% → 4% → 5%",
            "RV2": "3% → 3.5% → 4%",
            "RV3": "1% → 2%",
        },
        "Reward progression": {
            "RV1": "20% sucrose → 15%",
            "RV2": "10% → 15% → plain water → 5%; Oct 2026: 5% → 10%",
            "RV3": "5% → 10%",
        },
    }
    rows = [[name, *(context[name][v] for v in village_order)] for name in context]

    summary_metrics = [
        ("Average trials per day per subject", "Trials per rat per trial day", "All trials"),
        ("Average sessions per day per subject", "Sessions per rat per trial day", "Sessions"),
        ("Working Memory", "Trials per rat per trial day by experiment", "Working Memory"),
        ("Probabilistic Inference", "Trials per rat per trial day by experiment", "Probabilistic Inference"),
        ("Cognitive Bias", "Trials per rat per trial day by experiment", "Cognitive Bias"),
        ("Trials in Summer", "Trials per rat per trial day by season", "Summer"),
        ("Trials in Winter", "Trials per rat per trial day by season", "Winter"),
        ("Trials in Autumn", "Trials per rat per trial day by season", "Autumn"),
        ("Trials in Spring", "Trials per rat per trial day by season", "Spring"),
    ]
    for label, metric, category in summary_metrics:
        rows.append([
            label,
            *(_comparison_stat(subject_rates, v, metric, category) for v in village_order),
        ])

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Difference / measure", *village_order],
        colWidths=[0.31, 0.23, 0.23, 0.23],
        cellLoc="left",
        loc="center",
        bbox=[0, 0.08, 1, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1F3A5F")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2:
            cell.set_facecolor("#F3F6F9")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#D1D5DB")
    fig.suptitle("Village characteristics and full-period averages",
                 fontsize=15, weight="bold")
    fig.text(0.01, 0.025,
             "Average entries show mean [minimum–maximum] across rats, with n as the number "
             "of rats contributing. Each rat's value is its total divided by its distinct trial days.",
             fontsize=8, color="#4B5563")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    concentration_rows = []
    for level in ["20%", "15%", "10%", "5%"]:
        concentration_rows.append([
            f"Sucrose reward {level}",
            *(_comparison_stat(
                subject_rates, v,
                "Trials per rat per trial day by sucrose concentration", level,
            ) for v in village_order),
        ])
    for level in ["3%", "5%", "1%"]:
        concentration_rows.append([
            f"CA water {level}",
            *(_comparison_stat(
                subject_rates, v,
                "Trials per rat per trial day by CA water concentration", level,
            ) if not (v == "RV2" and level == "5%") else "Not recorded in diary"
              for v in village_order),
        ])

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis("off")
    table = ax.table(
        cellText=concentration_rows,
        colLabels=["Intervention period", *village_order],
        colWidths=[0.31, 0.23, 0.23, 0.23],
        cellLoc="left",
        loc="center",
        bbox=[0, 0.15, 1, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1F3A5F")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2:
            cell.set_facecolor("#F3F6F9")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#D1D5DB")
    fig.suptitle("Average trials during documented reward and CA concentrations",
                 fontsize=15, weight="bold")
    fig.text(0.01, 0.04,
             "Values show mean [minimum–maximum] across rats, n=number of rats. "
             "Transition dates without a recorded time are excluded. RV2's 5% sucrose "
             "value uses only the clearly dated 20–22 Oct 2026 window; the earlier 14-Nov-2025 "
             "5% entry has no recorded stop date. RV2 CA water was recorded as 3%→3.5%; "
             "project notes later list 4%, with no 5% CA period recorded.",
             fontsize=8, color="#4B5563", wrap=True)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def _nice_trial_tick_step(maximum):
    """Choose a readable y-axis interval for a trial-rate chart."""
    if pd.isna(maximum) or maximum <= 10:
        return 1
    if maximum <= 20:
        return 2
    if maximum <= 50:
        return 5
    if maximum <= 100:
        return 10
    if maximum <= 500:
        return 50
    return 100


def _annotate_interventions(ax, layout):
    """Mark intervention dates; their descriptions are listed above the plot."""
    for event_date, _, _, _ in layout:
        ax.axvline(event_date, color="#B45309", linestyle="--", linewidth=1.0,
                   alpha=0.78, zorder=1)


def _combined_intervention_layout(weekly):
    """Lay out intervention dates across the shared timeline."""
    dates_to_labels = {}
    for village, events in INTERVENTION_EVENTS.items():
        for date, label in events:
            event_date = pd.Timestamp(date)
            dates_to_labels.setdefault(event_date, []).append(f"{village}: {label}")

    x_start = pd.to_datetime(weekly["week_start"]).min()
    x_end = pd.to_datetime(weekly["week_start"]).max() + pd.Timedelta(days=6)
    if dates_to_labels:
        x_start = min(x_start, min(dates_to_labels))
        x_end = max(x_end, max(dates_to_labels))
    if x_end <= x_start:
        x_end = x_start + pd.Timedelta(days=1)

    width = 0.085
    gap = 0.01
    lane_right_edges = []
    layout = []
    span = (x_end - x_start).total_seconds()
    for event_date, labels in sorted(dates_to_labels.items()):
        x_fraction = (event_date - x_start).total_seconds() / span
        lane = next(
            (i for i, right_edge in enumerate(lane_right_edges)
             if x_fraction - width / 2 >= right_edge + gap),
            None,
        )
        if lane is None:
            lane = len(lane_right_edges)
            lane_right_edges.append(-np.inf)
        lane_right_edges[lane] = x_fraction + width / 2
        layout.append((event_date, "\n".join(labels), x_fraction, lane))
    return x_start, x_end, layout, len(lane_right_edges)


def _set_full_timeline_ticks(ax):
    """Use quarterly labels so the complete multi-year timeline stays legible."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


def _shade_timeline_seasons(ax, x_start, x_end):
    """Shade meteorological seasons across every year in a shared timeline."""
    for year in range(x_start.year, x_end.year + 1):
        _shade_seasons(ax, year)
    ax.set_xlim(x_start, x_end)


def _draw_overall_trials_comparison(pdf, weekly):
    """Compare the weekly overall trial rate across villages on one timeline."""
    village_order = ["RV1", "RV2", "RV3"]
    palette = {"RV1": "#0072B2", "RV2": "#D55E00", "RV3": "#009E73"}
    timeline_start, timeline_end, layout, _ = _combined_intervention_layout(weekly)
    fig, ax = plt.subplots(figsize=(18, 13))

    for village in village_order:
        subset = weekly.loc[weekly["village"].eq(village)].sort_values("week_start")
        if subset.empty:
            continue
        ax.plot(
            subset["week_start"],
            pd.to_numeric(subset["mean_trials_per_subject_day"], errors="coerce"),
            color=palette[village], linewidth=2.0, marker="o", markersize=2.8,
            label=village, zorder=3,
        )

    data_max = pd.to_numeric(
        weekly["mean_trials_per_subject_day"], errors="coerce"
    ).max(skipna=True)
    ax.set_xlim(timeline_start, timeline_end)
    _annotate_interventions(ax, layout)
    ax.set_ylim(0, 170)
    _shade_timeline_seasons(ax, timeline_start, timeline_end)
    ax.set_ylabel("Trials per rat per observed trial day")
    ax.set_xlabel("Week starting Monday")
    ax.yaxis.set_major_locator(MultipleLocator(_nice_trial_tick_step(data_max)))
    _set_full_timeline_ticks(ax)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)

    from matplotlib.lines import Line2D
    village_handles = [
        Line2D([0], [0], color=palette[v], linewidth=2, label=v)
        for v in village_order
    ]
    fig.legend(
        handles=village_handles + _season_handles(),
        loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=4, frameon=False, fontsize=9,
    )
    fig.suptitle(
        "Weekly average trials per rat across Rat Villages",
        fontsize=15, weight="bold", y=0.99,
    )
    fig.text(
        0.04, 0.91,
        "Interventions (vertical dashed lines mark event dates)",
        fontsize=10, weight="bold", color="#374151",
    )

    # Put readable event notes in a dedicated band above the chart so they do
    # not cover weekly values or force a larger y-axis range.
    events = sorted(
        (pd.Timestamp(date), village, label)
        for village, village_events in INTERVENTION_EVENTS.items()
        for date, label in village_events
    )
    columns = [events[i::3] for i in range(3)]
    event_colors = {"RV1": palette["RV1"], "RV2": palette["RV2"],
                    "RV3": palette["RV3"]}
    column_lefts = (0.04, 0.365, 0.69)
    column_width = 0.27
    panel_bottom, panel_top = 0.615, 0.89
    for left, entries in zip(column_lefts, columns):
        fig.add_artist(Rectangle(
            (left, panel_bottom), column_width, panel_top-panel_bottom,
            transform=fig.transFigure, facecolor="white",
            edgecolor="#D1D5DB", linewidth=0.7, zorder=0,
        ))
        row_step = (panel_top - panel_bottom - 0.018) / max(len(entries), 1)
        for row_index, (event_date, village, label) in enumerate(entries):
            y = panel_top - 0.018 - row_index * row_step
            fig.text(
                left + 0.008, y,
                f"{event_date:%d %b %Y}  {village}: {label}",
                fontsize=7.7, color=event_colors[village],
                va="top", ha="left", clip_on=True,
            )

    # Reserve the top band for intervention notes and keep the data plot at
    # the requested 0–170 scale.
    ax.set_position([0.055, 0.075, 0.925, 0.515])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _draw_experiment_comparison(pdf, weekly, experiment_weekly):
    """Compare each task family across villages on shared full-period axes."""
    village_order = ["RV1", "RV2", "RV3"]
    palette = {"RV1": "#0072B2", "RV2": "#D55E00", "RV3": "#009E73"}
    timeline_start = pd.to_datetime(weekly["week_start"]).min()
    timeline_end = pd.to_datetime(weekly["week_start"]).max() + pd.Timedelta(days=6)
    intervention_dates = sorted({
        pd.Timestamp(date)
        for events in INTERVENTION_EVENTS.values()
        for date, _ in events
    })

    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharex=True, sharey=True, squeeze=False)
    for ax, experiment in zip(axes.flat, EXPERIMENT_FAMILIES):
        subset = experiment_weekly.loc[
            experiment_weekly["experiment"].eq(experiment)
        ]
        for village in village_order:
            series = subset.loc[subset["village"].eq(village)].sort_values("week_start")
            if series.empty:
                continue
            ax.plot(
                series["week_start"],
                pd.to_numeric(series["mean_trials_per_subject_day"], errors="coerce"),
                color=palette[village], linewidth=1.7, marker="o", markersize=2.2,
                label=village, zorder=3,
            )
        for event_date in intervention_dates:
            ax.axvline(event_date, color="#9A3412", linestyle=":", linewidth=0.55,
                       alpha=0.22, zorder=1)
        _shade_timeline_seasons(ax, timeline_start, timeline_end)
        ax.set_ylim(0, 150)
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.set_title(experiment, fontsize=11, weight="bold")
        ax.set_ylabel("Trials per rat per observed trial day", fontsize=8)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        _set_full_timeline_ticks(ax)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=palette[v], linewidth=2, label=v)
        for v in village_order
    ] + _season_handles()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97),
               ncol=4, frameon=False, fontsize=8)
    fig.suptitle(
        "Weekly trial averages by experiment across Rat Villages",
        fontsize=14, weight="bold", y=0.995,
    )
    for ax in axes[-1, :]:
        ax.set_xlabel("Week starting Monday")
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.92))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _draw_category_period_summary(pdf, summary):
    """Draw one full-period table comparing task categories across villages."""
    village_order = ["RV1", "RV2", "RV3"]
    values = summary.set_index(["experiment", "village"])[
        "mean_trials_per_subject_day"
    ] if not summary.empty else pd.Series(dtype=float)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    fig.suptitle(
        "Average trials per rat per day by task category",
        fontsize=15,
        weight="bold",
        y=0.96,
    )

    if summary.empty:
        ax.text(0.5, 0.5, "No classified task trials available",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
    else:
        rows = []
        for category in EXPERIMENT_FAMILIES:
            row = [category]
            for village in village_order:
                value = values.get((category, village), np.nan)
                row.append(f"{value:,.1f}" if pd.notna(value) else "No data")
            rows.append(row)
        table = ax.table(
            cellText=rows,
            colLabels=["Task category", *village_order],
            cellLoc="center",
            colLoc="center",
            colWidths=[0.34, 0.22, 0.22, 0.22],
            bbox=[0.04, 0.30, 0.92, 0.42],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (row_index, _), cell in table.get_celld().items():
            cell.set_edgecolor("#D1D5DB")
            if row_index == 0:
                cell.set_facecolor("#E8EEF8")
                cell.set_text_props(weight="bold", color="#1F2937")
            else:
                cell.set_facecolor("white")

        ax.text(
            0.66,
            0.77,
            "Mean trials per active rat-day",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
            color="#1F2937",
            transform=ax.transAxes,
        )
        ax.text(
            0.02,
            0.16,
            "For each rat, category trials were divided by that rat's number "
            "of distinct dates with trials in the category. The table shows "
            "the equal-weighted mean across rats with at least one trial in "
            "that category during the period.",
            ha="left",
            va="top",
            fontsize=9,
            color="#4B5563",
            transform=ax.transAxes,
            wrap=True,
        )

    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _biweekly_measure_summary():
    """Build fortnightly body weight, intake, temperature, and humidity means."""
    workbook = Path(settings.RAT_HEALTH_FILE)
    anchor = pd.Timestamp(settings.BIWEEKLY_PERIOD_ANCHOR)

    def add_period(frame, date_column="date"):
        frame = frame.copy()
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
        frame = frame.dropna(subset=[date_column])
        offset_days = (frame[date_column] - anchor).dt.days
        frame["period_start"] = anchor + pd.to_timedelta(
            (offset_days // 14) * 14, unit="D"
        )
        return frame

    # Weight sheets are wide: animal IDs are rows and weigh dates are columns.
    weight_parts = []
    for village_number, sheet_name in settings.RAT_WEIGHT_SHEETS.items():
        frame = pd.read_excel(workbook, sheet_name=sheet_name)
        frame.columns = [str(column).strip() if column is not None else "" for column in frame.columns]
        rat_column = next((column for column in frame if column.casefold() == "rat_id"), None)
        date_map = {}
        for column in frame.columns:
            if column == rat_column:
                continue
            parsed_date = pd.to_datetime(column, errors="coerce")
            if pd.notna(parsed_date):
                date_map[column] = parsed_date.normalize()
        if rat_column is None or not date_map:
            LOGGER.warning("No dated weight measurements found in %s.", sheet_name)
            continue
        long = frame.melt(
            id_vars=[rat_column], value_vars=list(date_map),
            var_name="measurement_date", value_name="weight_g",
        ).rename(columns={rat_column: "subject"})
        long["date"] = long["measurement_date"].map(date_map)
        long["weight_g"] = pd.to_numeric(long["weight_g"], errors="coerce")
        long["subject"] = long["subject"].astype("string").str.strip().str.lower()
        long["village"] = f"RV{village_number}"
        long = add_period(long).dropna(subset=["weight_g", "subject"])
        if not long.empty:
            # Average repeated weigh-ins within rat and fortnight first, so
            # rats with more weigh-ins do not dominate the village mean.
            per_rat = long.groupby(
                ["village", "period_start", "subject"], as_index=False
            )["weight_g"].mean()
            weight_parts.append(per_rat)
    if weight_parts:
        weight = pd.concat(weight_parts, ignore_index=True)
        weight_summary = weight.groupby(
            ["village", "period_start"], as_index=False
        ).agg(body_weight_g=("weight_g", "mean"), weight_rats=("subject", "nunique"))
    else:
        weight_summary = pd.DataFrame(columns=["village", "period_start", "body_weight_g", "weight_rats"])

    # The workbook stores these daily village totals and per-rat values.
    food_water = pd.read_excel(workbook, sheet_name=settings.DAILY_FOOD_WATER_SHEET)
    food_water.columns = [str(column).strip().lower() for column in food_water.columns]
    required_food = {"ratvillage", "date"}
    if not required_food.issubset(food_water.columns):
        raise ValueError(
            f"{settings.DAILY_FOOD_WATER_SHEET} must contain ratvillage and date columns."
        )
    village_numbers = pd.to_numeric(food_water["ratvillage"], errors="coerce")
    food_water["village"] = "RV" + village_numbers.astype("Int64").astype("string")
    food_water["water_ml_per_rat"] = pd.to_numeric(
        food_water.get("water_ml_per_rat"), errors="coerce"
    )
    food_water["food_g_per_rat"] = pd.to_numeric(
        food_water.get("food_g_per_rat"), errors="coerce"
    )
    if "water_ml_per_rat" not in food_water or food_water["water_ml_per_rat"].isna().all():
        if {"water_ml", "rats_alive"}.issubset(food_water.columns):
            food_water["water_ml_per_rat"] = (
                pd.to_numeric(food_water["water_ml"], errors="coerce")
                / pd.to_numeric(food_water["rats_alive"], errors="coerce").replace(0, np.nan)
            )
    if "food_g_per_rat" not in food_water or food_water["food_g_per_rat"].isna().all():
        if {"food_g", "rats_alive"}.issubset(food_water.columns):
            food_water["food_g_per_rat"] = (
                pd.to_numeric(food_water["food_g"], errors="coerce")
                / pd.to_numeric(food_water["rats_alive"], errors="coerce").replace(0, np.nan)
            )
    food_water = add_period(food_water)
    food_water = food_water.loc[food_water["village"].isin(["RV1", "RV2", "RV3"])]
    daily_intake = food_water.groupby(["village", "date"], as_index=False).agg(
        ca_water_ml_per_rat_day=("water_ml_per_rat", "mean"),
        food_g_per_rat_day=("food_g_per_rat", "mean"),
        period_start=("period_start", "first"),
    )
    intake_summary = daily_intake.groupby(
        ["village", "period_start"], as_index=False
    ).agg(
        ca_water_ml_per_rat_day=("ca_water_ml_per_rat_day", "mean"),
        water_days=("ca_water_ml_per_rat_day", "count"),
        food_g_per_rat_day=("food_g_per_rat_day", "mean"),
        food_days=("food_g_per_rat_day", "count"),
    )

    # Environment logs are reduced to a daily mean across available sensor
    # channels, then averaged within each 14-day period. For a village without
    # a sensor log, use the midpoint of its recorded daily high and low values.
    from prepare_data import _load_environment_data

    environment_parts = []
    fallback_villages = []
    for village_number in (1, 2, 3):
        environment = _load_environment_data(village_number)
        if environment is not None:
            temp_columns = [
                column for column in environment.columns
                if column.casefold().endswith("_c") and column.casefold() != "control_c"
            ]
            humidity_columns = [
                column for column in environment.columns
                if column.casefold().endswith("_rh")
            ]
            if "__env_time_utc" not in environment:
                continue
            local_time = environment["__env_time_utc"].dt.tz_convert(settings.REPORT_TIMEZONE)
            environment["date"] = local_time.dt.tz_localize(None).dt.normalize()
            environment["temperature_c"] = (
                environment[temp_columns].mean(axis=1) if temp_columns else np.nan
            )
            environment["humidity_pct"] = (
                environment[humidity_columns].mean(axis=1) if humidity_columns else np.nan
            )
            daily_environment = environment.groupby("date", as_index=False).agg(
                temperature_c=("temperature_c", "mean"),
                humidity_pct=("humidity_pct", "mean"),
            )
        else:
            sheet_name = settings.FOOD_WATER_SHEETS.get(village_number)
            if not sheet_name:
                continue
            legacy = pd.read_excel(workbook, sheet_name=sheet_name)
            legacy.columns = [str(column).strip().lower() for column in legacy.columns]
            required = {"date_start", "temp_highest", "temp_lowest", "humidity_highest", "humidity_lowest"}
            if not required.issubset(legacy.columns):
                continue
            legacy["date"] = pd.to_datetime(legacy["date_start"], errors="coerce").dt.normalize()
            for column in ("temp_highest", "temp_lowest", "humidity_highest", "humidity_lowest"):
                legacy[column] = pd.to_numeric(legacy[column], errors="coerce")
            daily_environment = legacy.assign(
                temperature_c=legacy[["temp_highest", "temp_lowest"]].mean(axis=1),
                humidity_pct=legacy[["humidity_highest", "humidity_lowest"]].mean(axis=1),
            ).groupby("date", as_index=False).agg(
                temperature_c=("temperature_c", "mean"),
                humidity_pct=("humidity_pct", "mean"),
            )
            fallback_villages.append(f"RV{village_number}")
        daily_environment["village"] = f"RV{village_number}"
        daily_environment = add_period(daily_environment)
        environment_parts.append(daily_environment)

    if environment_parts:
        environment_daily = pd.concat(environment_parts, ignore_index=True)
        environment_summary = environment_daily.groupby(
            ["village", "period_start"], as_index=False
        ).agg(
            temperature_c=("temperature_c", "mean"),
            temperature_days=("temperature_c", "count"),
            humidity_pct=("humidity_pct", "mean"),
            humidity_days=("humidity_pct", "count"),
        )
    else:
        environment_summary = pd.DataFrame(columns=[
            "village", "period_start", "temperature_c", "temperature_days",
            "humidity_pct", "humidity_days",
        ])

    summary = weight_summary.merge(
        intake_summary, on=["village", "period_start"], how="outer"
    ).merge(environment_summary, on=["village", "period_start"], how="outer")
    summary = summary.sort_values(["village", "period_start"]).reset_index(drop=True)
    if fallback_villages:
        notes = (
            "Temperature and humidity are daily means across sensor channels where available. "
            f"{', '.join(fallback_villages)} use the midpoint of the workbook's recorded daily "
            "high and low values because no merged sensor log is configured."
        )
    else:
        notes = "Temperature and humidity are daily means across the available sensor channels."
    return summary, notes


def _draw_biweekly_measure_plot(pdf, summary, source_note):
    """Plot biweekly weight, CA-water intake, food, temperature, and humidity."""
    village_order = ["RV1", "RV2", "RV3"]
    palette = {"RV1": "#0072B2", "RV2": "#D55E00", "RV3": "#009E73"}
    metrics = [
        ("body_weight_g", "Mean body weight (g)"),
        ("ca_water_ml_per_rat_day", "CA water intake (mL per rat per day)"),
        ("food_g_per_rat_day", "Food intake (g per rat per day)"),
        ("temperature_c", "Daily mean temperature (°C)"),
        ("humidity_pct", "Daily mean relative humidity (%)"),
    ]
    fig, axes = plt.subplots(5, 1, figsize=(17, 13), sharex=True, squeeze=False)
    axes = axes[:, 0]
    if summary.empty:
        axes[0].text(0.5, 0.5, "No weight, intake, or environment data available",
                     ha="center", va="center", transform=axes[0].transAxes)
    for ax, (column, label) in zip(axes, metrics):
        has_values = False
        for village in village_order:
            subset = summary.loc[
                summary["village"].eq(village)
                & pd.to_numeric(summary.get(column), errors="coerce").notna()
            ].sort_values("period_start")
            if subset.empty:
                continue
            has_values = True
            ax.plot(
                subset["period_start"], subset[column],
                color=palette[village], linewidth=1.6, marker="o", markersize=2.5,
                label=village,
            )
        if not has_values:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                    transform=ax.transAxes, color="#6B7280")
        ax.set_ylabel(label, fontsize=8)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axes[-1].set_xlabel("14-day period starting")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=palette[v], linewidth=2, label=v)
               for v in village_order]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Biweekly body weight, intake, temperature, and humidity",
                 fontsize=15, weight="bold", y=0.995)
    fig.text(0.01, 0.012,
             "Each point is a 14-day mean. Weight is averaged within rat first; "
             "intake is per rat per day. " + source_note,
             fontsize=7.5, color="#4B5563", wrap=True)
    fig.tight_layout(rect=(0.035, 0.055, 0.99, 0.92))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def generate(data):
    """Write weekly trial and occupancy graphs into one PDF plus CSV summaries."""
    required = {"ratvillage", "subject", "datetime"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Weekly report missing columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(data["datetime"]):
        data = data.copy()
        try:
            # Prepared CSVs can contain ISO timestamps with and without
            # fractional seconds. Ask pandas to parse each value separately.
            data["datetime"] = pd.to_datetime(
                data["datetime"], format="mixed", utc=True
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Weekly report could not parse the datetime column. "
                "Expected ISO timestamps, which may have mixed precision."
            ) from exc
    elif data["datetime"].dt.tz is None:
        raise ValueError("Weekly report requires timezone-aware datetime values.")

    frame = data.dropna(subset=["ratvillage", "subject", "datetime"]).copy()
    if frame.empty:
        return None, None
    numbers = frame["ratvillage"].astype(str).str.extract(r"([0-9]+)$", expand=False)
    if numbers.isna().any():
        raise ValueError("Could not determine Rat Village for every trial.")
    frame["village"] = "RV" + numbers
    frame["subject"] = frame["subject"].astype("string").str.strip().str.lower()
    aliases = {
        str(old).strip().lower(): str(new).strip().lower()
        for old, new in getattr(settings, "EVENT_SUBJECT_ALIASES", {}).items()
    }
    frame["subject"] = frame["subject"].replace(aliases)
    frame["local_datetime"] = frame["datetime"].dt.tz_convert(settings.REPORT_TIMEZONE)
    frame["day"] = frame["local_datetime"].dt.tz_localize(None).dt.normalize()
    frame["week_start"] = frame["day"] - pd.to_timedelta(
        frame["day"].dt.weekday, unit="D")

    roster = _living_roster(frame)
    matched = frame[["village", "subject", "day"]].drop_duplicates().merge(
        roster[["village", "subject", "day"]],
        on=["village", "subject", "day"], how="left",
        indicator=True, validate="one_to_one")
    mismatches = matched.loc[matched["_merge"].ne("both")]
    if not mismatches.empty:
        raise ValueError(
            "Trial records do not match a living rat in rat_details. "
            "Check animal aliases, start and death dates. Examples: "
            f"{mismatches[['village', 'subject', 'day']].head(8).to_dict('records')}"
        )

    daily = _calendar_and_daily(frame)
    # Reconstruct the living count from rat_details on every date, including
    # trial-free days when a death could make the carried-forward count stale.
    living = roster.groupby(["village", "day"]).size().rename("living_subjects")
    daily = daily.drop(columns=["living_subjects", "population_source"]).merge(
        living.reset_index(), on=["village", "day"], how="left",
        validate="one_to_one")
    daily["living_subjects"] = daily["living_subjects"].fillna(0).astype(int)
    daily["population_source"] = "rat_details"
    box = _box_minutes(frame)
    daily = daily.merge(box, on=["village", "day"], how="left")
    daily.loc[daily["total_trials"].eq(0), "box_minutes"] = 0
    daily["trial_hour_occupancy_pct"] = (
        daily["active_rat_hours"] / (24 * daily["living_subjects"].replace(0, np.nan)) * 100)
    daily["actual_box_occupancy_pct"] = (
        daily["box_minutes"] / (1440 * daily["living_subjects"].replace(0, np.nan)) * 100)
    daily["week_start"] = daily["day"] - pd.to_timedelta(
        daily["day"].dt.weekday, unit="D")
    # Keep the count of calendar days separate from dates with trial records.
    # Neither column is the rat-specific divisor in the trial averages.
    daily["day_with_trials"] = daily["total_trials"].gt(0)
    weekly = (
        daily.groupby(["village", "week_start"], as_index=False)
        .agg(total_trials=("total_trials", "sum"),
             active_rat_hours=("active_rat_hours", "sum"),
             box_minutes=("box_minutes", "sum"),
             subject_days=("living_subjects", "sum"),
             calendar_days=("day", "size"),
             observed_days=("day_with_trials", "sum"))
    )

    # Identical individual-rat calculation for both plotted measures.
    living_weeks = roster[["village", "subject", "week_start"]].drop_duplicates()
    rat_week = _subject_rates(living_weeks, frame, ["village", "week_start"])
    rat_week_summary = _summarise_rates(rat_week, ["village", "week_start"])
    weekly = weekly.drop(columns=["total_trials"]).merge(
        rat_week_summary, on=["village", "week_start"], how="left",
        validate="one_to_one")
    if weekly["mean_trials_per_subject_day"].isna().any():
        raise ValueError("Weekly trial rate without a living rat")

    frame["experiment"] = _experiment_family(frame)
    classified = frame.loc[frame["experiment"].isin(EXPERIMENT_FAMILIES)].copy()
    category_daily = (
        classified.groupby(["village", "experiment", "day"])
        .size().rename("total_trials").reset_index()
    )
    # Current source data has no explicit assignment history. Infer each
    # experiment's participant cohort from animals observed in that category
    # at least once. Include zero-trial weeks for these rats while alive.
    known_participants = classified[
        ["village", "experiment", "subject"]].drop_duplicates()
    active_week_parts = []
    for village, village_days in daily.groupby("village", sort=True):
        categories = [
            family for family in EXPERIMENT_FAMILIES
            if family in set(category_daily.loc[
                category_daily["village"].eq(village), "experiment"
            ])
        ]
        for experiment in categories:
            family_days = category_daily.loc[
                category_daily["village"].eq(village)
                & category_daily["experiment"].eq(experiment)
            ]
            active_days = village_days.loc[
                village_days["day"].between(
                    family_days["day"].min(), family_days["day"].max()), ["day"]
            ].copy()
            active_days["village"] = village
            active_days["experiment"] = experiment
            active_days["week_start"] = active_days["day"] - pd.to_timedelta(
                active_days["day"].dt.weekday, unit="D")
            active_week_parts.append(
                active_days.groupby(
                    ["village", "experiment", "week_start"], as_index=False
                ).agg(days_in_week=("day", "size"))
            )

    category_columns = [
        "village", "experiment", "week_start", "days_in_week",
        "total_trials", "subjects_with_trials", "eligible_subjects",
        "mean_trials_per_subject_day",
    ]
    if active_week_parts:
        active_weeks = pd.concat(active_week_parts, ignore_index=True)
        eligible_category = (
            active_weeks[["village", "experiment", "week_start"]]
            .merge(living_weeks, on=["village", "week_start"], how="inner",
                   validate="many_to_many")
            .merge(known_participants,
                   on=["village", "experiment", "subject"], how="inner",
                   validate="many_to_one")
        )
        category_rat_week = _subject_rates(
            eligible_category, classified,
            ["village", "experiment", "week_start"]
        )
        by_category = _summarise_rates(
            category_rat_week, ["village", "experiment", "week_start"]
        )
        experiment_weekly = active_weeks.merge(
            by_category, on=["village", "experiment", "week_start"],
            how="left", validate="one_to_one")
        for column in ("total_trials", "subjects_with_trials", "eligible_subjects",
                       "mean_trials_per_subject_day"):
            experiment_weekly[column] = experiment_weekly[column].fillna(0)
    else:
        category_rat_week = pd.DataFrame(columns=[
            "village", "experiment", "week_start", "subject",
            "total_trials", "unique_trial_days", "subject_daily_average"])
        experiment_weekly = pd.DataFrame(columns=category_columns)
    experiment_weekly = experiment_weekly[category_columns].sort_values(
        ["village", "experiment", "week_start"])


    # Full-period Method 1: calculate one rate per rat and category, then
    # average those rates equally across rats with trials in that category.
    period_summary_rows = []
    classified = frame.loc[frame["experiment"].isin(EXPERIMENT_FAMILIES)]
    for (village, experiment), category_frame in classified.groupby(
        ["village", "experiment"], sort=True
    ):
        period_start = category_frame["day"].min()
        period_end = category_frame["day"].max()
        rat_totals = category_frame.groupby("subject").size().rename("total_trials")
        rat_days = category_frame.groupby("subject")["day"].nunique().rename("observed_days")
        rat_summary = pd.concat([rat_totals, rat_days], axis=1)
        rat_summary["trials_per_active_day"] = (
            rat_summary["total_trials"] / rat_summary["observed_days"]
        )
        period_summary_rows.append({
            "village": village,
            "experiment": experiment,
            "period_start": period_start,
            "period_end": period_end,
            "rat_days_with_trials": int(rat_summary["observed_days"].sum()),
            "total_trials": int(rat_summary["total_trials"].sum()),
            "subjects_with_trials": int(rat_summary.shape[0]),
            "mean_trials_per_subject_day": float(rat_summary["trials_per_active_day"].mean()),
        })
    category_period_summary = pd.DataFrame(period_summary_rows, columns=[
        "village", "experiment", "period_start", "period_end", "rat_days_with_trials",
        "total_trials", "subjects_with_trials", "mean_trials_per_subject_day",
    ])
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

    # Overall and experiment summaries now use the same _subject_rates method.

    weekly = weekly.sort_values(["village", "week_start"])

    directory = Path(settings.REPORT_DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "Weekly_Daily_Trials.csv"
    weekly.to_csv(csv_path, index=False, encoding="utf-8-sig")
    rat_week.to_csv(directory / "Weekly_Daily_Trials_By_Subject.csv",
                    index=False, encoding="utf-8-sig")
    category_rat_week.to_csv(
        directory / "Weekly_Daily_Trials_By_Experiment_Subject.csv",
        index=False, encoding="utf-8-sig")
    experiment_csv_path = directory / "Weekly_Daily_Trials_By_Experiment.csv"
    experiment_weekly.to_csv(experiment_csv_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved weekly trial averages by experiment to %s", experiment_csv_path)
    category_summary_path = directory / "Trials_Per_Day_By_Experiment_Summary.csv"
    category_period_summary.to_csv(category_summary_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved full-period category averages to %s", category_summary_path)
    biweekly_measures, environment_source_note = _biweekly_measure_summary()
    biweekly_measure_path = directory / "Biweekly_Animal_Environment_Summary.csv"
    biweekly_measures.to_csv(biweekly_measure_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved biweekly animal and environment averages to %s", biweekly_measure_path)
    boxplot_subject_rates = _boxplot_subject_rates(frame)
    rat_details = pd.read_excel(
        settings.RAT_HEALTH_FILE,
        sheet_name=settings.RAT_DETAILS_SHEET,
    )
    boxplot_subject_path = directory / "Weekly_Daily_Trials_Boxplot_By_Subject.csv"
    boxplot_subject_rates.to_csv(boxplot_subject_path, index=False, encoding="utf-8-sig")
    boxplot_summary_path = directory / "Weekly_Daily_Trials_Boxplot_Summary.csv"
    _boxplot_summary(boxplot_subject_rates).to_csv(
        boxplot_summary_path, index=False, encoding="utf-8-sig")
    daily_path = directory / "Weekly_Daily_Occupancy.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    pdf_path = directory / "Weekly_Daily_Trials.pdf"
    with PdfPages(pdf_path) as pdf:
        _draw_overall_trials_comparison(pdf, weekly)
        _draw_experiment_comparison(pdf, weekly, experiment_weekly)
        _draw_category_period_summary(pdf, category_period_summary)
        _draw_biweekly_measure_plot(pdf, biweekly_measures, environment_source_note)
        _draw_boxplot_pages(pdf, boxplot_subject_rates)
        _draw_comparison_summary_pages(pdf, rat_details, boxplot_subject_rates)
    return pdf_path, csv_path


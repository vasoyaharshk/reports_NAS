"""Create the combined current-status and recent-activity PDF report."""
from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

import settings


STATUS_COLUMNS = [
    "ratvillage", "subject", "task", "stage", "substage", "block_number",
    "block_accuracy", "current_ror", "trial_counter_ror", "total_trials",
    "block_number_task", "N_trials_block", "N_trials_stage", "N_trials_substage",
]
CB_TASK = "Cognitive_Bias_Auditory_Training_60"
CB_PR_TASK = "Cognitive_Bias_Auditory_Training_60_PR"


def _truthy_zero_or_missing(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.eq(0) | numeric.isna()


def _date_and_time(data: pd.DataFrame) -> pd.DataFrame:
    """Create local report dates and timestamps from prepared UTC datetimes."""
    frame = data.copy()
    timezone = getattr(settings, "REPORT_TIMEZONE", "Europe/Madrid")
    if "datetime" in frame:
        timestamps = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        local = timestamps.dt.tz_convert(timezone)
        frame["_report_datetime"] = timestamps
        frame["_report_date"] = local.dt.date
        frame["_report_hour"] = local.dt.hour
    else:
        frame["_report_datetime"] = pd.NaT
        frame["_report_date"] = pd.to_datetime(
            frame.get("date", pd.Series(pd.NaT, index=frame.index)), errors="coerce"
        ).dt.date
        frame["_report_hour"] = np.nan
    if "_report_date" in frame and "date" in frame:
        fallback = pd.to_datetime(frame["date"], errors="coerce").dt.date
        frame["_report_date"] = frame["_report_date"].where(
            pd.notna(frame["_report_date"]), fallback
        )
    return frame


def latest_status_table(data: pd.DataFrame) -> pd.DataFrame:
    """Return each rat's latest eligible trial and its block/stage counts."""
    required = {"ratvillage", "subject", "task", "datetime", "trial_result"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Status report needs columns: {', '.join(missing)}")
    frame = _date_and_time(data)
    frame = frame.loc[frame["trial_result"].notna()].copy()
    frame = frame.loc[frame["task"].astype("string").ne("Automatic_Water")]
    if "forced_choice_actual_trial" in frame:
        frame = frame.loc[_truthy_zero_or_missing(frame["forced_choice_actual_trial"])]
    frame = frame.dropna(subset=["_report_datetime", "subject"])
    group_keys = ["ratvillage", "subject"]
    latest = (frame.sort_values("_report_datetime", kind="stable")
              .groupby(group_keys, dropna=False, sort=False).tail(1).copy())
    if "total_trials" in latest:
        latest["block_number_task"] = np.floor(
            pd.to_numeric(latest["total_trials"], errors="coerce") / 40
        )
    else:
        latest["block_number_task"] = np.nan

    count_keys = {
        "N_trials_block": [*group_keys, "task", "stage", "block_number"],
        "N_trials_stage": [*group_keys, "task", "stage"],
        "N_trials_substage": [*group_keys, "task", "substage"],
    }
    for output, keys in count_keys.items():
        keys = [key for key in keys if key in frame.columns and key in latest.columns]
        if len(keys) == len(count_keys[output]):
            counts = frame.groupby(keys, dropna=False).size().rename(output).reset_index()
            latest = latest.merge(counts, on=keys, how="left", validate="many_to_one")
        else:
            latest[output] = np.nan

    for column in STATUS_COLUMNS:
        if column not in latest:
            latest[column] = pd.NA
    latest = latest[STATUS_COLUMNS].sort_values(
        ["ratvillage", "task", "subject", "stage", "substage"],
        na_position="last", kind="stable",
    ).reset_index(drop=True)
    return latest


def recent_daily_village_metrics(data: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Count task trials and distinct trial-hours by village and local date."""
    required = {"ratvillage", "task", "datetime", "trial_result"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Activity report needs columns: {', '.join(missing)}")
    frame = _date_and_time(data)
    frame = frame.loc[frame["trial_result"].notna()].copy()
    if "subject" in frame:
        frame = frame.loc[frame["subject"].astype("string").str.lower().ne("joy")]
    frame = frame.dropna(subset=["ratvillage", "_report_date", "_report_datetime", "task"])
    if frame.empty:
        return pd.DataFrame(columns=["ratvillage", "date", "task", "trials", "active_hours", "occupancy_pct"])

    daily_trials = (frame.groupby(["ratvillage", "_report_date", "task"], dropna=False)
                    .size().rename("trials").reset_index())
    active_hours = (frame.groupby(["ratvillage", "_report_date"], dropna=False)
                    ["_report_hour"].nunique().rename("active_hours").reset_index())
    daily = daily_trials.merge(active_hours, on=["ratvillage", "_report_date"], how="left")
    daily["occupancy_pct"] = daily["active_hours"] / 24 * 100
    daily = daily.rename(columns={"_report_date": "date"})

    keep = {}
    for village, village_frame in daily.groupby("ratvillage", dropna=False):
        dates = sorted(village_frame["date"].dropna().unique())
        keep[village] = set(dates[-max(1, int(n_days)):])
    mask = [date in keep[village] for village, date in zip(daily["ratvillage"], daily["date"])]
    return daily.loc[mask].sort_values(["ratvillage", "date", "task"], kind="stable").reset_index(drop=True)


def _draw_status_pages(pdf: PdfPages, status: pd.DataFrame) -> None:
    if status.empty:
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No eligible trial records for current status.", ha="center", va="center")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return
    page_size = 24
    display_headers = {
        "ratvillage": "Village", "subject": "Subject", "task": "Task",
        "stage": "Stage", "substage": "Substage", "block_number": "Block",
        "block_accuracy": "Block acc.", "current_ror": "Current ROR",
        "trial_counter_ror": "ROR counter", "total_trials": "Total trials",
        "block_number_task": "Task block", "N_trials_block": "Block N",
        "N_trials_stage": "Stage N", "N_trials_substage": "Sub-N",
    }
    column_widths = [0.045, 0.07, 0.20, 0.045, 0.06, 0.055, 0.075,
                     0.075, 0.075, 0.065, 0.065, 0.065, 0.065, 0.04]
    for page_num, start in enumerate(range(0, len(status), page_size), start=1):
        subset = status.iloc[start:start + page_size].copy().fillna("—")
        subset["task"] = subset["task"].astype(str).map(
            lambda value: "\n".join(textwrap.wrap(value, width=25, break_long_words=True, break_on_hyphens=False))
        )
        for column in ("block_accuracy", "current_ror"):
            subset[column] = pd.to_numeric(subset[column], errors="coerce").map(
                lambda value: "—" if pd.isna(value) else f"{value:.3f}"
            )
        for column in ("trial_counter_ror", "total_trials", "block_number_task",
                       "N_trials_block", "N_trials_stage", "N_trials_substage"):
            subset[column] = pd.to_numeric(subset[column], errors="coerce").map(
                lambda value: "—" if pd.isna(value) else str(int(value))
            )
        fig, ax = plt.subplots(figsize=(16.5, 11.7))
        ax.axis("off")
        ax.set_title("Latest Training Status - All Rat Villages", fontsize=17, weight="bold", pad=24)
        ax.text(0.5, 1.005, f"Rows {start + 1}-{start + len(subset)} of {len(status)} · Page {page_num}",
                ha="center", va="bottom", fontsize=9, transform=ax.transAxes)
        if start == 0 and getattr(settings, "REPORT_DATA_NOTE", ""):
            ax.text(0.5, 0.97, settings.REPORT_DATA_NOTE, ha="center", va="top",
                    fontsize=9, color="#9a3412", transform=ax.transAxes)
        table = ax.table(cellText=subset.astype(str).values,
                         colLabels=[display_headers[column] for column in subset.columns],
                         colWidths=column_widths, cellLoc="center", colLoc="center",
                         bbox=[0.02, 0.10, 0.96, 0.70])
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1, 1.45)
        for (row, _), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#214761")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#edf3f6")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _draw_daily_village_pages(pdf: PdfPages, daily: pd.DataFrame) -> None:
    if daily.empty:
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No dated trial rows available for recent activity.", ha="center", va="center")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return
    for village, rows in daily.groupby("ratvillage", dropna=False, sort=True):
        trials = rows.pivot_table(index="date", columns="task", values="trials", aggfunc="sum", fill_value=0)
        occupancy = rows.groupby("date", as_index=True)["occupancy_pct"].first().reindex(trials.index)
        fig, (ax_trials, ax_occupancy) = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True,
                                                     gridspec_kw={"height_ratios": [1.5, 1]})
        trials.plot(kind="bar", stacked=True, ax=ax_trials, width=0.82, colormap="tab20")
        ax_trials.set_title(f"Recent Trial Activity — Rat Village {village}", fontsize=15, weight="bold")
        ax_trials.set_ylabel("Trials per day")
        ax_trials.grid(axis="y", alpha=0.22)
        ax_trials.legend(title="Task", loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
        ax_occupancy.bar(range(len(occupancy)), occupancy.values, color="#36a386", width=0.78)
        ax_occupancy.set_ylim(0, 100)
        ax_occupancy.set_ylabel("Trial-hour occupancy (%)")
        ax_occupancy.set_xlabel("Date")
        ax_occupancy.set_yticks(range(0, 101, 20))
        ax_occupancy.grid(axis="y", alpha=0.22)
        ax_occupancy.set_xticks(range(len(occupancy)))
        ax_occupancy.set_xticklabels([pd.Timestamp(day).strftime("%d %b") for day in occupancy.index],
                                     rotation=45, ha="right", fontsize=8)
        fig.text(0.01, 0.01, "Occupancy = distinct local clock hours containing trials ÷ 24 × 100.", fontsize=8, color="#555555")
        fig.tight_layout(rect=(0, 0.04, 0.83, 1))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _cb_dataset(data: pd.DataFrame, task_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = data.copy()
    task = frame["task"].astype("string")
    frame.loc[task.isin(["Cognitive_Bias_Auditory_Training_60R", CB_TASK]), "task"] = CB_TASK
    frame = frame.loc[frame["task"].astype("string").eq(task_name)].copy()
    required = {"ratvillage", "subject", "group", "pair", "block_number", "tone_played",
                "forced_choice_actual_trial", "trial_result", "touchoutside"}
    if not required.issubset(frame.columns):
        return frame.iloc[0:0], frame.iloc[0:0], frame.iloc[0:0]
    frame["group_pair_label"] = (
        "Group " + frame["group"].astype("string") + " | Pair " + frame["pair"].astype("string")
    )
    frame["_tone"] = frame["tone_played"].astype("string").str.lower().map({"low": "Low", "high": "High"}).fillna("Unknown")
    frame["_fc"] = pd.to_numeric(frame["forced_choice_actual_trial"], errors="coerce")
    result = frame["trial_result"].astype("string").str.lower()
    if "success" in frame:
        frame["_success"] = pd.to_numeric(frame["success"], errors="coerce")
    else:
        frame["_success"] = result.map({"correct": 1.0, "correct_first": 1.0, "incorrect": 0.0})
    valid = result.isin(["correct", "incorrect"]) & frame["_fc"].eq(0)
    keys = ["ratvillage", "subject", "group", "pair", "block_number", "_tone"]
    summary = (frame.loc[valid].groupby(keys, dropna=False).agg(
        success=("_success", "mean"), free_n=("trial_result", "size")
    ).reset_index())
    forced = (frame.loc[frame["_fc"].eq(1)].groupby(keys, dropna=False).size().rename("forced_n").reset_index())
    touch = pd.to_numeric(frame["touchoutside"], errors="coerce").eq(1)
    touch_counts = (frame.assign(_touch=touch.astype(int)).groupby(keys, dropna=False)['_touch']
                    .sum().rename("touchoutside_n").reset_index())
    incorrect = (frame.loc[result.eq("incorrect")].groupby(keys, dropna=False)
                .size().rename("incorrect_n").reset_index())
    counts = summary.merge(forced, how="left", on=keys).merge(touch_counts, how="left", on=keys).merge(incorrect, how="left", on=keys)
    for column in ("forced_n", "touchoutside_n", "incorrect_n"):
        counts[column] = counts[column].fillna(0)
    counts["block_total_n"] = counts["free_n"] + counts["forced_n"]
    counts["forced_pct_total"] = counts["forced_n"] / counts["block_total_n"].replace(0, np.nan)
    counts["touchoutside_pct_incorrect"] = counts["touchoutside_n"] / counts["incorrect_n"].replace(0, np.nan)
    counts["group_pair_label"] = "Group " + counts["group"].astype("string") + " | Pair " + counts["pair"].astype("string")
    line_keys = ["ratvillage", "subject", "group", "pair", "block_number", "group_pair_label"]
    lines = counts.groupby(line_keys, dropna=False).agg(
        forced_n_total=("forced_n", "sum"), block_total_n_total=("block_total_n", "sum"),
        touchoutside_n_total=("touchoutside_n", "sum"), incorrect_n_total=("incorrect_n", "sum"),
    ).reset_index()
    lines["forced_pct_total"] = lines["forced_n_total"] / lines["block_total_n_total"].replace(0, np.nan)
    lines["touchoutside_pct_incorrect"] = lines["touchoutside_n_total"] / lines["incorrect_n_total"].replace(0, np.nan)
    return frame, counts, lines


def _draw_cb_pages(pdf: PdfPages, data: pd.DataFrame) -> None:
    if not bool(getattr(settings, "Cognitive_bias", True)) or "task" not in data:
        return
    frame = data.copy()
    task_names = frame["task"].astype("string")
    frame.loc[task_names.eq("Cognitive_Bias_Auditory_Training_60R"), "task"] = CB_TASK
    configs = [
        (2, CB_TASK, None, {"chand"}, ""),
        (3, CB_TASK, None, set(), ""),
        (2, CB_PR_TASK, {"fergus"}, {"chand"}, " PR"),
    ]
    for village, task_name, selected, excluded, suffix in configs:
        task_frame, summary, lines = _cb_dataset(frame.loc[pd.to_numeric(frame["ratvillage"], errors="coerce").eq(village)], task_name)
        if summary.empty:
            continue
        subjects = sorted(summary["subject"].dropna().astype(str).unique())
        subjects = [subject for subject in subjects if subject.lower() not in excluded]
        if selected is not None:
            subjects = [subject for subject in subjects if subject.lower() in selected]
        for set_no, offset in enumerate(range(0, len(subjects), 2), start=1):
            chosen = subjects[offset:offset + 2]
            sub_summary = summary[summary["subject"].astype(str).isin(chosen)]
            sub_lines = lines[lines["subject"].astype(str).isin(chosen)]
            facets = sorted(sub_summary["group_pair_label"].astype(str).unique())
            if not facets:
                continue
            fig, axes = plt.subplots(len(facets), len(chosen), figsize=(12, max(4, 3.1 * len(facets))), squeeze=False, sharey=True)
            fig.suptitle(f"Cognitive Bias{suffix} · RV{village} · Set {set_no}", fontsize=15, weight="bold")
            for row_idx, facet in enumerate(facets):
                for col_idx, subject in enumerate(chosen):
                    ax = axes[row_idx, col_idx]
                    rows = sub_summary[(sub_summary["group_pair_label"].astype(str).eq(facet)) & (sub_summary["subject"].astype(str).eq(subject))]
                    lrows = sub_lines[(sub_lines["group_pair_label"].astype(str).eq(facet)) & (sub_lines["subject"].astype(str).eq(subject))]
                    blocks = sorted(rows["block_number"].dropna().unique(), key=lambda value: str(value))
                    x = np.arange(len(blocks))
                    width = 0.36
                    for tone_idx, tone in enumerate(("Low", "High")):
                        values = rows.loc[rows["_tone"].eq(tone)].set_index("block_number")["success"].reindex(blocks)
                        ax.bar(x + (tone_idx - 0.5) * width, values, width=width,
                               color=("#0072B2" if tone == "Low" else "#E69F00"), label=tone)
                    line_lookup = lrows.drop_duplicates("block_number").set_index("block_number").reindex(blocks)
                    ax.plot(x, line_lookup["forced_pct_total"], color="#009E73", linewidth=1.4, marker="o", label="Forced %")
                    ax.plot(x, line_lookup["touchoutside_pct_incorrect"], color="#CC79A7", linewidth=1.4,
                            linestyle="dotted", marker="o", label="Outside %")
                    ax.axhline(0.7, color="#444444", linewidth=0.8)
                    ax.axhline(0.5, color="#777777", linewidth=0.8, linestyle="dotted")
                    ax.set_ylim(0, 1)
                    ax.set_yticks(np.arange(0, 1.01, 0.2), labels=[f"{int(v*100)}%" for v in np.arange(0, 1.01, 0.2)])
                    ax.set_xticks(x, [str(block) for block in blocks], rotation=90, fontsize=7)
                    ax.grid(axis="y", alpha=0.2)
                    ax.set_title(subject if row_idx == 0 else "", fontsize=10)
                    if col_idx == 0:
                        ax.set_ylabel(f"{facet}\nProportion", fontsize=8)
                    if row_idx == len(facets) - 1:
                        ax.set_xlabel("Block")
                    # Legend uses the same five series as the R plot.
                    if row_idx == 0 and col_idx == len(chosen) - 1:
                        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
            fig.tight_layout(rect=(0, 0, 0.88, 0.95))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def generate(data: pd.DataFrame) -> Path:
    """Write status, daily trials, occupancy, and Cognitive Bias pages to one PDF."""
    output_dir = Path(settings.REPORT_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "Status_and_Activity_Report.pdf"
    status = latest_status_table(data)
    daily = recent_daily_village_metrics(data, getattr(settings, "STATUS_REPORT_DAYS", 30))
    with PdfPages(output_path) as pdf:
        _draw_status_pages(pdf, status)
        _draw_daily_village_pages(pdf, daily)
        _draw_cb_pages(pdf, data)
    return output_path

"""Shared helpers for the task-specific PDF reports."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

import settings


def read_sorted_data(path: Path | None = None) -> pd.DataFrame:
    """Read the complete, semicolon-delimited output from prepare_data.py."""
    source = Path(path or (Path(settings.OUTPUT_DIRECTORY) / settings.OUTPUT_FILENAME))
    if not source.is_file():
        raise FileNotFoundError(
            f"Prepared data not found: {source}. Run main.py or prepare_data.py first."
        )
    return pd.read_csv(source, sep=settings.CSV_SEPARATOR, low_memory=False)


def task_family(data: pd.DataFrame, family: str) -> pd.DataFrame:
    """Select task rows using the category assigned by prepare_data.py."""
    if "task_category" in data.columns:
        return data.loc[data["task_category"].eq(family)].copy()
    # Keep report modules usable with older sorted files.
    task = data.get("task", pd.Series("", index=data.index)).astype("string").str.lower()
    if family == "working_memory":
        mask = task.str.contains(r"working.?memory|\bwm\b", regex=True, na=False)
    elif family == "cognitive_bias":
        mask = task.str.contains(r"cognitive.?bias|^cb", regex=True, na=False)
    else:
        mask = task.str.contains(r"probability|weber|^pi", regex=True, na=False)
    return data.loc[mask].copy()


def _session_summary(data: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("ratvillage", "subject", "task", "stage", "substage", "session") if c in data]
    if not keys:
        return pd.DataFrame()
    frame = data.copy()
    if "success" not in frame and "trial_result" in frame:
        result = frame["trial_result"].astype("string").str.lower()
        frame["success"] = result.isin(["correct", "correct_first"]).astype(float)
        frame.loc[~result.isin(["correct", "correct_first", "incorrect"]), "success"] = float("nan")
    if "trial_result" in frame:
        valid = frame["trial_result"].astype("string").str.lower().ne("miss") & frame["trial_result"].notna()
    else:
        valid = pd.Series(True, index=frame.index)
    frame["_valid"] = valid.astype(int)
    success_values = frame["success"] if "success" in frame else pd.Series(float("nan"), index=frame.index)
    frame["_success"] = pd.to_numeric(success_values, errors="coerce")
    result = frame.groupby(keys, dropna=False).agg(
        total_trials=("_valid", "size"),
        valid_trials=("_valid", "sum"),
        accuracy=("_success", "mean"),
    ).reset_index()
    return result


def _add_page_title(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    ax.axis("off")
    ax.text(0.5, 0.97, title, ha="center", va="top", fontsize=20, weight="bold", transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.91, subtitle, ha="center", va="top", fontsize=10, color="#555555", transform=ax.transAxes)


def write_task_report(data: pd.DataFrame, family: str, title: str, output_path: Path) -> Path:
    """Create a compact PDF summary with overview tables and per-village plots."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sessions = _session_summary(data)
    tasks = sorted(data.get("task", pd.Series(dtype="string")).dropna().astype(str).unique())
    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        _add_page_title(ax, title, f"Generated from {len(data):,} trial records")
        if data.empty:
            ax.text(0.5, 0.52, "No rows match this task family.", ha="center", va="center", fontsize=14, transform=ax.transAxes)
        else:
            if "date" in data:
                dates = pd.to_datetime(data["date"], errors="coerce").dropna()
                date_range = f"Date range: {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}" if len(dates) else "Date range unavailable"
            else:
                date_range = "Date range unavailable"
            ax.text(0.06, 0.82, date_range, fontsize=11, transform=ax.transAxes)
            ax.text(0.06, 0.77, f"Task names: {len(tasks)}", fontsize=11, transform=ax.transAxes)
            ax.text(0.06, 0.72, f"Subjects: {data['subject'].nunique() if 'subject' in data else 'unknown'}", fontsize=11, transform=ax.transAxes)
            if not sessions.empty:
                summary_keys = [c for c in ("ratvillage", "task") if c in sessions]
                summary = sessions.groupby(summary_keys, dropna=False).agg(
                    sessions=("valid_trials", "size"),
                    valid_trials=("valid_trials", "sum"),
                    mean_accuracy=("accuracy", "mean"),
                ).reset_index()
                summary["mean_accuracy"] = (summary["mean_accuracy"] * 100).round(1)
                summary = summary.sort_values(summary_keys)
                table = ax.table(
                    cellText=summary.head(24).fillna("—").values,
                    colLabels=[*summary.columns[:-1], "Mean accuracy (%)"],
                    cellLoc="center", colLoc="center", bbox=[0.06, 0.12, 0.88, 0.5],
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.2)
                if len(summary) > 24:
                    ax.text(0.06, 0.08, "Overview table shows the first 24 village/task rows.", fontsize=8, color="#555555", transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if not sessions.empty:
            villages: Iterable[object] = sorted(sessions["ratvillage"].dropna().unique()) if "ratvillage" in sessions else [None]
            for village in villages:
                selected = sessions if village is None else sessions.loc[sessions["ratvillage"].eq(village)]
                if selected.empty:
                    continue
                fig, axes = plt.subplots(2, 1, figsize=(11.7, 8.3), gridspec_kw={"height_ratios": [1.4, 1]})
                rv_label = "All villages" if village is None else f"Rat Village {village}"
                axes[0].set_title(f"{title}: valid trials per session — {rv_label}", fontsize=14, weight="bold")
                if "session" in selected and "subject" in selected:
                    selected = selected.copy()
                    selected["session_label"] = selected["session"].astype("string")
                    # Limit the page to a readable set of subjects; data tables retain all.
                    subjects = selected["subject"].dropna().astype(str).unique()
                    for subject in subjects[:18]:
                        subject_rows = selected.loc[selected["subject"].astype(str).eq(subject)].sort_values("session_label")
                        axes[0].plot(subject_rows["session_label"], subject_rows["valid_trials"], marker="o", linewidth=1, label=subject)
                    axes[0].set_xlabel("Session")
                    axes[0].set_ylabel("Valid trials")
                    axes[0].grid(axis="y", alpha=0.25)
                    if len(subjects) <= 18:
                        axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, title="Subject")
                    else:
                        axes[0].text(0.99, 0.98, f"Showing {min(18, len(subjects))} of {len(subjects)} subjects", ha="right", va="top", transform=axes[0].transAxes, fontsize=8)
                else:
                    axes[0].text(0.5, 0.5, "Session or subject field unavailable", ha="center", va="center", transform=axes[0].transAxes)
                    axes[0].axis("off")
                accuracy = selected.groupby("task", dropna=False)["accuracy"].mean().dropna().sort_values(ascending=False) if "task" in selected else pd.Series(dtype=float)
                axes[1].set_title("Mean session accuracy by task", fontsize=11)
                if accuracy.empty:
                    axes[1].text(0.5, 0.5, "Accuracy field unavailable", ha="center", va="center", transform=axes[1].transAxes)
                    axes[1].axis("off")
                else:
                    accuracy.plot(kind="bar", ax=axes[1], color="#3b82a0")
                    axes[1].set_ylim(0, 1)
                    axes[1].set_ylabel("Accuracy")
                    axes[1].set_xlabel("")
                    axes[1].tick_params(axis="x", labelrotation=35, labelsize=8)
                    axes[1].grid(axis="y", alpha=0.25)
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    return output_path


def enabled_task_data(data: pd.DataFrame, family: str, setting_name: str) -> pd.DataFrame | None:
    if not bool(getattr(settings, setting_name)):
        return None
    return task_family(data, family)

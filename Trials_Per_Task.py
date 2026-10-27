"""Create task trial-count tables and a daily-count PDF from all sorted rows."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

import settings


def _daily_counts(data: pd.DataFrame) -> pd.DataFrame:
    required = {"ratvillage", "subject", "task", "date"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Trials-per-task report needs columns: {', '.join(missing)}")
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame = frame.dropna(subset=["date", "task", "subject"])
    return (frame.groupby(["ratvillage", "subject", "task", "date"], dropna=False)
            .size().rename("trials").reset_index())


def generate(data: pd.DataFrame) -> tuple[Path, Path]:
    """Always process complete data; task switches are intentionally ignored."""
    output_dir = Path(settings.REPORT_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily = _daily_counts(data)
    csv_path = output_dir / "Trials_Per_Task_Daily.csv"
    daily.to_csv(csv_path, index=False, encoding="utf-8-sig")
    pdf_path = output_dir / "Trials_Per_Task_Report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        ax.text(0.5, 0.95, "Trials per Task", ha="center", va="top", fontsize=20, weight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.89, f"All task families included · {len(data):,} sorted trial rows", ha="center", va="top", fontsize=10, transform=ax.transAxes)
        if daily.empty:
            ax.text(0.5, 0.5, "No dated trial rows available.", ha="center", va="center", fontsize=13, transform=ax.transAxes)
        else:
            summary = daily.groupby(["ratvillage", "task"], dropna=False).agg(
                days=("date", "nunique"), total_trials=("trials", "sum"),
                mean_trials_per_day=("trials", "mean"),
            ).reset_index().sort_values(["ratvillage", "task"])
            summary["mean_trials_per_day"] = summary["mean_trials_per_day"].round(1)
            table = ax.table(cellText=summary.head(28).values, colLabels=summary.columns,
                             cellLoc="center", colLoc="center", bbox=[0.05, 0.12, 0.9, 0.68])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            if len(summary) > 28:
                ax.text(0.05, 0.08, "First 28 village/task rows shown. Full totals are in the daily CSV.", fontsize=8, transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if not daily.empty:
            daily_village = daily.groupby(["ratvillage", "task", "date"], dropna=False)["trials"].sum().reset_index()
            latest = max(daily_village["date"])
            cutoff = pd.Timestamp(latest) - pd.Timedelta(days=settings.TRIALS_REPORT_DAYS - 1)
            daily_village["date_dt"] = pd.to_datetime(daily_village["date"])
            recent = daily_village.loc[daily_village["date_dt"].ge(cutoff)]
            for village, group in recent.groupby("ratvillage", dropna=False):
                fig, ax = plt.subplots(figsize=(11.7, 8.3))
                for task, task_data in group.groupby("task", dropna=False):
                    total = task_data.groupby("date_dt")["trials"].sum().sort_index()
                    ax.plot(total.index, total.values, marker="o", linewidth=1.5, label=str(task))
                ax.set_title(f"Daily trials by task · Rat Village {village}", fontsize=15, weight="bold")
                ax.set_xlabel("Date")
                ax.set_ylabel("Trials")
                ax.grid(alpha=0.25)
                ax.tick_params(axis="x", rotation=35)
                ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, title="Task")
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    return pdf_path, csv_path

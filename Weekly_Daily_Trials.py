"""Weekly daily trial statistics and plots for each Rat Village."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import pandas as pd

import settings


def generate(data):
    """Export one PDF page per village and the underlying weekly statistics."""
    required = {"ratvillage", "subject", "datetime"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Weekly trials missing columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(data["datetime"]) or data["datetime"].dt.tz is None:
        raise ValueError("Weekly trials requires timezone-aware parsed datetime.")

    frame = data.loc[data["datetime"].notna() & data["subject"].notna()].copy()
    if frame.empty:
        return None, None

    village_numbers = frame["ratvillage"].astype(str).str.extract(r"([0-9]+)$", expand=False)
    if village_numbers.isna().any():
        raise ValueError("Cannot determine village number for some trial rows.")
    frame["village"] = "RV" + village_numbers
    frame["day"] = (
        frame["datetime"]
        .dt.tz_convert(settings.REPORT_TIMEZONE)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    daily = (
        frame.groupby(["village", "subject", "day"], as_index=False)
        .size()
        .rename(columns={"size": "trials"})
    )

    # Count zero-trial days between a subject's first and last observed
    # trial dates. Dates outside this span are not assumed to be active.
    spans = (
        daily.groupby(["village", "subject"])["day"]
        .agg(first_day="min", last_day="max")
        .reset_index()
    )
    calendar = []
    for row in spans.itertuples(index=False):
        calendar.append(pd.DataFrame({
            "village": row.village,
            "subject": row.subject,
            "day": pd.date_range(row.first_day, row.last_day, freq="D"),
        }))

    complete = pd.concat(calendar, ignore_index=True).merge(
        daily, on=["village", "subject", "day"], how="left", validate="one_to_one"
    )
    complete["trials"] = complete["trials"].fillna(0).astype(int)
    complete["week_start"] = (
        complete["day"] - pd.to_timedelta(complete["day"].dt.weekday, unit="D")
    )
    weekly = (
        complete.groupby(["village", "week_start"], as_index=False)
        .agg(
            mean_trials_per_subject_day=("trials", "mean"),
            min_trials_per_subject_day=("trials", "min"),
            max_trials_per_subject_day=("trials", "max"),
            subject_days=("trials", "size"),
            total_trials=("trials", "sum"),
        )
        .sort_values(["village", "week_start"])
    )

    directory = Path(settings.REPORT_DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "Weekly_Daily_Trials.csv"
    weekly.to_csv(csv_path, index=False, encoding="utf-8-sig")

    pdf_path = directory / "Weekly_Daily_Trials.pdf"
    with PdfPages(pdf_path) as pdf:
        for village, subset in weekly.groupby("village", sort=True):
            subset = subset.sort_values("week_start")
            # Scale width with the number of weeks so all vertical labels fit.
            fig_width = max(13, 0.38 * len(subset) + 3)
            fig, ax = plt.subplots(figsize=(fig_width, 7))
            dates = subset["week_start"]
            mean = subset["mean_trials_per_subject_day"].to_numpy()
            minimum = subset["min_trials_per_subject_day"].to_numpy()
            maximum = subset["max_trials_per_subject_day"].to_numpy()

            ax.fill_between(
                dates, minimum, maximum,
                color="#6495ed", alpha=0.28, label="Minimum–maximum band"
            )
            ax.plot(
                dates, maximum, color="#d97706", linestyle="--",
                marker="^", markersize=3, linewidth=1.8, label="Maximum"
            )
            ax.plot(
                dates, mean, color="#1e3a8a",
                marker="o", markersize=3, linewidth=2.5, label="Average"
            )
            ax.plot(
                dates, minimum, color="#059669", linestyle="--",
                marker="v", markersize=3, linewidth=1.8, label="Minimum"
            )
            ax.set_xticks(dates)
            ax.set_xticklabels(
                [date.strftime("%d %b %Y") for date in dates],
                rotation=90, ha="center", fontsize=8
            )
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.set_ylim(bottom=0)
            ax.set_title(f"{village}: Weekly daily trials per rat")
            ax.set_xlabel("Week starting Monday")
            ax.set_ylabel("Trials per rat per day")
            ax.grid(alpha=0.2)
            ax.legend(ncol=4, loc="upper center")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path, csv_path

"""Weekly per-subject daily trial statistics for every Rat Village."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import settings

def generate(data):
    required = {"ratvillage", "subject", "datetime"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Weekly trials missing columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(data["datetime"]) or data["datetime"].dt.tz is None:
        raise ValueError("Weekly trials requires timezone-aware parsed datetime.")
    frame = data.loc[data["datetime"].notna() & data["subject"].notna()].copy()
    frame["village"] = "RV" + frame["ratvillage"].astype(str).str.extract(r"(\\d+)$", expand=False)
    frame["day"] = frame["datetime"].dt.tz_convert(settings.REPORT_TIMEZONE).dt.tz_localize(None).dt.normalize()
    frame["week_start"] = frame["day"] - pd.to_timedelta(frame["day"].dt.weekday, unit="D")
    daily = frame.groupby(["village", "subject", "day"], as_index=False).size().rename(columns={"size": "trials"})
    # A subject is counted from their first observed day to their last observed day.
    # Without independent roster dates, days outside that interval cannot be assumed active.
    spans = daily.groupby(["village", "subject"])["day"].agg(["min", "max"]).reset_index()
    pieces = []
    for row in spans.itertuples(index=False):
        days = pd.date_range(row.min, row.max, freq="D")
        pieces.append(pd.DataFrame({"village": row.village, "subject": row.subject, "day": days}))
    if not pieces:
        return None, None
    complete = pd.concat(pieces, ignore_index=True).merge(daily, on=["village", "subject", "day"], how="left")
    complete["trials"] = complete["trials"].fillna(0).astype(int)
    complete["week_start"] = complete["day"] - pd.to_timedelta(complete["day"].dt.weekday, unit="D")
    weekly = complete.groupby(["village", "week_start"], as_index=False).agg(
        mean_trials_per_subject_day=("trials", "mean"),
        min_trials_per_subject_day=("trials", "min"),
        max_trials_per_subject_day=("trials", "max"),
        subject_days=("trials", "size"),
        total_trials=("trials", "sum"),
    )
    directory = Path(settings.REPORT_DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "Weekly_Daily_Trials.csv"
    weekly.to_csv(csv_path, index=False)
    pdf_path = directory / "Weekly_Daily_Trials.pdf"
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pdf_path) as pdf:
        for village, subset in weekly.groupby("village", sort=True):
            fig, ax = plt.subplots(figsize=(13, 6))
            subset = subset.sort_values("week_start")
            for column, label, color in [
                ("max_trials_per_subject_day", "Maximum", "#d97706"),
                ("mean_trials_per_subject_day", "Mean", "#2563eb"),
                ("min_trials_per_subject_day", "Minimum", "#059669"),
            ]:
                ax.plot(subset["week_start"], subset[column], label=label, color=color, linewidth=1.7)
            ax.set(title=f"{village}: Weekly daily trials per subject", xlabel="Week starting Monday",
                   ylabel="Trials per subject per day")
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return pdf_path, csv_path

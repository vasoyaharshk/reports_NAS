"""Regression checks for weekly unique-day rates and seasonal shading.

Run from repository root:
    python -m unittest test_weekly_daily_trials.py
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import Weekly_Daily_Trials as report


class WeeklyReportTests(unittest.TestCase):
    def test_unique_dates_and_zero_trial_rats(self):
        monday = pd.Timestamp("2026-10-05")
        keys = ["village", "week_start"]
        eligible = pd.DataFrame({
            "village": ["RV2"] * 3,
            "week_start": [monday] * 3,
            "subject": ["rat_a", "rat_b", "rat_c"],
        })
        trials = pd.DataFrame({
            "village": ["RV2"] * 8,
            "week_start": [monday] * 8,
            "subject": ["rat_a"] * 6 + ["rat_b"] * 2,
            "day": ([pd.Timestamp("2026-10-05")] * 3
                    + [pd.Timestamp("2026-10-07")] * 3
                    + [pd.Timestamp("2026-10-06")] * 2),
        })
        rates = report._subject_rates(eligible, trials, keys).set_index("subject")
        self.assertEqual(rates.loc["rat_a", "unique_trial_days"], 2)
        self.assertEqual(rates.loc["rat_a", "subject_daily_average"], 3.0)
        self.assertEqual(rates.loc["rat_b", "subject_daily_average"], 2.0)
        self.assertEqual(rates.loc["rat_c", "total_trials"], 0)
        self.assertEqual(rates.loc["rat_c", "subject_daily_average"], 0)
        summary = report._summarise_rates(rates.reset_index(), keys).iloc[0]
        self.assertEqual(summary["eligible_subjects"], 3)
        self.assertEqual(summary["subjects_with_trials"], 2)
        self.assertAlmostEqual(summary["mean_trials_per_subject_day"], 5 / 3)

    def test_same_function_for_experiment_with_zero_week(self):
        monday = pd.Timestamp("2026-10-05")
        keys = ["village", "experiment", "week_start"]
        eligible = pd.DataFrame({
            "village": ["RV2", "RV2"],
            "experiment": ["Cognitive Bias", "Cognitive Bias"],
            "week_start": [monday, monday],
            "subject": ["rat_a", "rat_b"],
        })
        trials = pd.DataFrame({
            "village": ["RV2"] * 5,
            "experiment": ["Cognitive Bias"] * 5,
            "week_start": [monday] * 5,
            "subject": ["rat_a"] * 5,
            "day": [pd.Timestamp("2026-10-06")] * 5,
        })
        rates = report._subject_rates(eligible, trials, keys)
        summary = report._summarise_rates(rates, keys).iloc[0]
        self.assertAlmostEqual(summary["mean_trials_per_subject_day"], 2.5)
        self.assertEqual(summary["eligible_subjects"], 2)
        self.assertEqual(summary["subjects_with_trials"], 1)

    def test_roster_uses_start_and_death_dates(self):
        frame = pd.DataFrame({
            "village": ["RV1", "RV1"],
            "day": [pd.Timestamp("2026-10-05"), pd.Timestamp("2026-10-08")],
        })
        details = pd.DataFrame({
            "ratvillage_number": [1, 1],
            "rat_id": ["rat_a", "rat_b"],
            "exp_start_date": ["2026-10-06", "2026-10-05"],
            "passed_away_date": ["2026-10-08", None],
        })
        with tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp:
            with patch.object(report.settings, "RAT_HEALTH_FILE", Path(tmp.name)):
                with patch.object(report.pd, "read_excel", return_value=details):
                    roster = report._living_roster(frame)
        a_dates = set(roster.loc[roster["subject"].eq("rat_a"), "day"])
        b_dates = set(roster.loc[roster["subject"].eq("rat_b"), "day"])
        self.assertEqual(a_dates, {
            pd.Timestamp("2026-10-06"), pd.Timestamp("2026-10-07")
        })
        self.assertEqual(len(b_dates), 4)

    def test_seasons_do_not_change_plot_limits(self):
        fig, ax = plt.subplots()
        try:
            ax.plot(pd.to_datetime(["2026-01-05", "2026-12-21"]), [1, 2])
            before = ax.get_xlim()
            report._shade_seasons(ax, 2026)
            self.assertEqual(len(ax.patches), 5)
            self.assertEqual(ax.get_xlim(), before)
            # Winter appears at both ends; each other season appears once.
            colors = [p.get_facecolor()[:3] for p in ax.patches]
            self.assertEqual(colors[0], colors[-1])
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()

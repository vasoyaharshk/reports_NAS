"""Shared living-population denominators for village-level reports."""
import logging
import pandas as pd

LOGGER = logging.getLogger(__name__)

def daily_population(data, timezone="Europe/Madrid"):
    """Return one population per village/local day, using an explicit field when present.

    Until the living-subjects field exists, infer provisional first/last trial
    spans. Never sum the repeated population column across trial rows.
    """
    required = {"ratvillage", "subject", "datetime"}
    if not required.issubset(data.columns):
        raise ValueError(f"Population requires {sorted(required)}")
    frame = data.dropna(subset=list(required)).copy()
    frame["day"] = pd.to_datetime(frame["datetime"], utc=True).dt.tz_convert(timezone).dt.date
    frame["village"] = frame["ratvillage"].astype(str).str.extract(r"(\d+)$", expand=False).astype(int)
    if "number_of_living_subjects" in frame.columns:
        frame["population"] = pd.to_numeric(frame["number_of_living_subjects"], errors="coerce")
        grouped = frame.groupby(["village", "day"])["population"]
        if grouped.nunique(dropna=True).gt(1).any():
            raise ValueError("Conflicting number_of_living_subjects within a village/day")
        result = grouped.first().reset_index()
        if result["population"].isna().any() or result["population"].le(0).any() or result["population"].mod(1).ne(0).any():
            raise ValueError("number_of_living_subjects must be a positive integer")
        result["population_source"] = "number_of_living_subjects"
    else:
        LOGGER.warning("Using provisional observed subject spans until number_of_living_subjects is available")
        spans = frame.groupby(["village", "subject"])["day"].agg(["min", "max"]).reset_index()
        pieces = []
        for row in spans.itertuples(index=False):
            pieces.append(pd.DataFrame({"village": row.village, "day": pd.date_range(row.min, row.max).date}))
        result = (pd.concat(pieces, ignore_index=True).groupby(["village", "day"])
                  .size().rename("population").reset_index())
        result["population_source"] = "provisional_observed_span"
    return result

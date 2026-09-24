"""Validated data loading for the AC One fuel-consumption experiment.

The source CSV contains two logically separate datasets:

* external-model forecasts, which are valid predictor inputs; and
* realised consumption, which is evaluation-only ground truth.

Forecasts and actuals are stored as paired scaled columns
(``forecast_minmax``/``actual_minmax`` and ``forecast_indexed``/``actual_indexed``).
There is no unsuffixed ``forecast`` or ``actual`` column. Each scale is a
separate experiment input and is scored only against its matching actual.

``horizon_date`` is the month the forecasting run happened, so it *is* the
forecast origin. Every row sharing a ``horizon_date`` also shares one
``model_version`` and one ``schd_file_id``, because a single run emits all
leads at once. The month a row forecasts is therefore derived rather than
stored -- see :func:`target_month_for`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.features import StaticFrameAdapter


DEFAULT_DATA_PATH = Path(__file__).with_name("VECTOR___AGENTIC_FORECASTING_DATA.csv")

#: Months between a reference month and the earliest date its realised
#: consumption could be known. A month's total is not complete until the month
#: ends, so a run at the start of month ``M`` knows actuals only through
#: ``M - 1``. One month is the earliest defensible lag; raise it if the real
#: reporting lag is longer.
ACTUALS_RELEASE_LAG_MONTHS = 1

ForecastScale = Literal["minmax", "indexed"]
FORECAST_SCALES: tuple[ForecastScale, ...] = ("minmax", "indexed")
SCALE_COLUMNS: dict[ForecastScale, tuple[str, str]] = {
    "minmax": ("forecast_minmax", "actual_minmax"),
    "indexed": ("forecast_indexed", "actual_indexed"),
}

REQUIRED_COLUMNS = {
    "station",
    "region",
    "horizon_date",
    "month_horizon",
    "forecast_minmax",
    "actual_minmax",
    "forecast_indexed",
    "actual_indexed",
    "schd_file_id",
    "model_version",
}

PREDICTOR_INPUT_COLUMNS = [
    "station",
    "region",
    "horizon_date",
    "month_horizon",
    "forecast_minmax",
    "forecast_indexed",
    "schd_file_id",
    "model_version",
    "forecast_origin",
    "target_month",
]


def target_month_for(origin: object, month_horizon: int) -> pd.Timestamp:
    """Return the month a run at ``origin`` forecasts at ``month_horizon``.

    A lead of 2 issued in January targets March. This matches the harness
    convention that horizon ``h`` means ``h`` frequency units past the origin,
    so ``as_of + offset * h`` stays equivalent.
    """
    return pd.Timestamp(origin) + pd.DateOffset(months=int(month_horizon))


def normalize_forecast_scale(scale: str) -> ForecastScale:
    """Return a validated forecast-scale identifier."""
    if scale not in FORECAST_SCALES:
        raise ValueError(f"Unknown forecast scale {scale!r}; expected one of {FORECAST_SCALES}.")
    return scale  # type: ignore[return-value]


def forecast_column(scale: str) -> str:
    """Return the predictor-input forecast column for a scale."""
    return SCALE_COLUMNS[normalize_forecast_scale(scale)][0]


def actual_column(scale: str) -> str:
    """Return the evaluation-only actual column for a scale."""
    return SCALE_COLUMNS[normalize_forecast_scale(scale)][1]


def station_series_id(station: str, scale: str) -> str:
    """Return the canonical actual-series ID for a station and forecast scale."""
    normalized = station.strip().lower().replace(" ", "_")
    return f"ac_one_fuel_consumption_{normalized}_{normalize_forecast_scale(scale)}"


def load_forecast_data(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load, validate, and enrich the anonymized forecast export.

    Returns a row per station, run month, and lead, with ``forecast_origin``
    (the run month) and ``target_month`` (the month being forecast) as added
    month-start columns. Duplicate keys, actuals that disagree for the same
    target month, non-positive horizons, and missing values fail loudly rather
    than silently changing the evaluation sample.
    """
    frame = pd.read_csv(path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"AC One dataset is missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("AC One dataset is empty.")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        null_columns = sorted(frame.columns[frame.isna().any()].tolist())
        raise ValueError(f"AC One dataset contains null values in: {null_columns}")

    frame = frame.copy()
    frame["horizon_date"] = pd.to_datetime(frame["horizon_date"], errors="raise")
    if not frame["horizon_date"].dt.is_month_start.all():
        raise ValueError("Every horizon_date must be a month-start timestamp.")

    frame["month_horizon"] = pd.to_numeric(frame["month_horizon"], errors="raise").astype(int)
    if (frame["month_horizon"] < 1).any():
        raise ValueError("month_horizon values must be positive integers.")

    numeric_columns = [
        "forecast_minmax",
        "actual_minmax",
        "forecast_indexed",
        "actual_indexed",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    key = ["station", "horizon_date", "month_horizon"]
    duplicate_rows = frame.duplicated(key, keep=False)
    if duplicate_rows.any():
        duplicate_keys = frame.loc[duplicate_rows, key].to_dict(orient="records")
        raise ValueError(f"AC One dataset contains duplicate forecast keys: {duplicate_keys}")

    frame["forecast_origin"] = frame["horizon_date"]
    frame["target_month"] = [
        target_month_for(origin, horizon)
        for origin, horizon in zip(frame["horizon_date"], frame["month_horizon"], strict=True)
    ]

    # The same target month is forecast by several runs at different leads.
    # Those rows must carry one realised value, or the evaluation sample
    # depends on which row a join happens to pick.
    for scale in FORECAST_SCALES:
        column = actual_column(scale)
        actual_counts = frame.groupby(["station", "target_month"])[column].nunique()
        inconsistent = actual_counts[actual_counts > 1]
        if not inconsistent.empty:
            raise ValueError(
                f"{column} must agree across every row sharing a station and target month: "
                f"{[(station, str(month.date())) for station, month in inconsistent.index]}"
            )

    return frame.sort_values(["station", "forecast_origin", "month_horizon"]).reset_index(drop=True)


def actuals_frame(
    data: pd.DataFrame,
    station: str,
    scale: str,
    *,
    release_lag_months: int = ACTUALS_RELEASE_LAG_MONTHS,
) -> pd.DataFrame:
    """Return one canonical actual observation per target month for a station/scale.

    ``released_at`` is stamped ``release_lag_months`` after the reference month.
    Without it the cutoff fence would hand a run at the start of month ``M`` the
    actual for ``M`` itself, which cannot be known until the month has ended.
    """
    resolved_scale = normalize_forecast_scale(scale)
    column = actual_column(resolved_scale)
    if "target_month" not in data.columns:
        raise ValueError("Normalized AC One data is missing the derived 'target_month' column.")
    station_rows = data.loc[data["station"] == station, ["target_month", column]]
    if station_rows.empty:
        raise KeyError(f"Unknown AC One station: {station!r}")

    frame = (
        station_rows.drop_duplicates()
        .rename(columns={"target_month": "timestamp", column: "value"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if frame["timestamp"].duplicated().any():
        duplicated = sorted({str(ts.date()) for ts in frame.loc[frame["timestamp"].duplicated(), "timestamp"]})
        raise ValueError(f"Conflicting {column} values for station {station!r} at target months: {duplicated}")

    frame["released_at"] = frame["timestamp"] + pd.DateOffset(months=release_lag_months)
    return frame


def forecast_input_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy containing predictor-safe columns and no outcomes."""
    missing = sorted(set(PREDICTOR_INPUT_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Normalized AC One data is missing predictor columns: {missing}")
    return data[PREDICTOR_INPUT_COLUMNS].copy()


def build_ac_one_service(
    data: pd.DataFrame | None = None,
    *,
    release_lag_months: int = ACTUALS_RELEASE_LAG_MONTHS,
) -> DataService:
    """Register one actual series per anonymized station and scale, keyed by target month."""
    loaded = load_forecast_data() if data is None else data.copy()
    service = DataService()
    for station in sorted(loaded["station"].unique()):
        for scale in FORECAST_SCALES:
            series_id = station_series_id(str(station), scale)
            service.register(
                series_id,
                StaticFrameAdapter(actuals_frame(loaded, str(station), scale, release_lag_months=release_lag_months)),
                SeriesMetadata(
                    series_id=series_id,
                    description=(f"Monthly airline fuel consumption at {station} ({scale} scale, anonymized)"),
                    source="AC One anonymized evaluation export",
                    units="anonymized minmax units" if scale == "minmax" else "anonymized indexed units",
                    frequency="MS",
                    table_id=DEFAULT_DATA_PATH.name,
                ),
            )
    return service


__all__ = [
    "ACTUALS_RELEASE_LAG_MONTHS",
    "DEFAULT_DATA_PATH",
    "FORECAST_SCALES",
    "PREDICTOR_INPUT_COLUMNS",
    "REQUIRED_COLUMNS",
    "SCALE_COLUMNS",
    "ForecastScale",
    "actual_column",
    "actuals_frame",
    "build_ac_one_service",
    "forecast_column",
    "forecast_input_frame",
    "load_forecast_data",
    "normalize_forecast_scale",
    "station_series_id",
    "target_month_for",
]

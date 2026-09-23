"""Validated data loading for the AC One fuel-consumption experiment.

The source CSV contains two logically separate datasets:

* external-model forecasts, which are valid predictor inputs; and
* realised consumption, which is evaluation-only ground truth.

Forecasts and actuals are stored as paired scaled columns
(``forecast_minmax``/``actual_minmax`` and ``forecast_indexed``/``actual_indexed``).
There is no unsuffixed ``forecast`` or ``actual`` column. Each scale is a
separate experiment input and is scored only against its matching actual.

Forecast origins are reconstructed as ``horizon_date - month_horizon`` months
because the anonymized export stores the target month rather than an issue
timestamp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.features import StaticFrameAdapter


DEFAULT_DATA_PATH = Path(__file__).with_name("VECTOR___AGENTIC_FORECASTING_DATA.csv")

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
]


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

    Returns a row per station, target month, and horizon with an additional
    ``forecast_origin`` month-start column. Duplicate keys, inconsistent
    actuals across horizons, non-positive horizons, and missing values fail
    loudly rather than silently changing the evaluation sample.
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

    for scale in FORECAST_SCALES:
        column = actual_column(scale)
        actual_counts = frame.groupby(["station", "horizon_date"])[column].nunique()
        inconsistent = actual_counts[actual_counts > 1]
        if not inconsistent.empty:
            raise ValueError(f"{column} must agree across horizons for each station/month: {list(inconsistent.index)}")

    frame["forecast_origin"] = [
        target - pd.DateOffset(months=int(horizon))
        for target, horizon in zip(frame["horizon_date"], frame["month_horizon"], strict=True)
    ]
    return frame.sort_values(["station", "forecast_origin", "month_horizon"]).reset_index(drop=True)


def actuals_frame(data: pd.DataFrame, station: str, scale: str) -> pd.DataFrame:
    """Return one canonical actual observation per target month for a station/scale."""
    resolved_scale = normalize_forecast_scale(scale)
    column = actual_column(resolved_scale)
    station_rows = data.loc[data["station"] == station, ["horizon_date", column]]
    if station_rows.empty:
        raise KeyError(f"Unknown AC One station: {station!r}")
    return (
        station_rows.drop_duplicates()
        .rename(columns={"horizon_date": "timestamp", column: "value"})
        .assign(released_at=lambda x: x["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def forecast_input_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy containing predictor-safe columns and no outcomes."""
    missing = sorted(set(PREDICTOR_INPUT_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Normalized AC One data is missing predictor columns: {missing}")
    return data[PREDICTOR_INPUT_COLUMNS].copy()


def build_ac_one_service(data: pd.DataFrame | None = None) -> DataService:
    """Register one evaluation-only actual series per anonymized station and scale."""
    loaded = load_forecast_data() if data is None else data.copy()
    service = DataService()
    for station in sorted(loaded["station"].unique()):
        for scale in FORECAST_SCALES:
            series_id = station_series_id(str(station), scale)
            service.register(
                series_id,
                StaticFrameAdapter(actuals_frame(loaded, str(station), scale)),
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
]

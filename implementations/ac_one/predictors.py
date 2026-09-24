"""Use-case predictors for the AC One forecast-adjustment experiment."""

from __future__ import annotations

from datetime import datetime, timezone
from math import isclose

import pandas as pd
from ac_one.data import (
    FORECAST_SCALES,
    ForecastScale,
    forecast_column,
    forecast_input_frame,
    normalize_forecast_scale,
    station_series_id,
)
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation import (
    STANDARD_QUANTILES,
    ContinuousForecast,
    ForecastingTask,
    Prediction,
    Predictor,
)


def station_for_task(task: ForecastingTask, data: pd.DataFrame) -> str:
    """Resolve a task target-series ID to exactly one station."""
    matches = [
        str(station)
        for station in data["station"].unique()
        for scale in FORECAST_SCALES
        if station_series_id(str(station), scale) == task.target_series_id
    ]
    unique_matches = sorted(set(matches))
    if len(unique_matches) != 1:
        raise ValueError(
            f"Could not map target series {task.target_series_id!r} to exactly one station; found {unique_matches}."
        )
    return unique_matches[0]


def scale_for_task(task: ForecastingTask) -> ForecastScale:
    """Parse the forecast scale encoded in a task's target-series ID."""
    series_id = task.target_series_id
    for scale in FORECAST_SCALES:
        if series_id.endswith(f"_{scale}"):
            return scale
    raise ValueError(f"Could not parse forecast scale from target series {series_id!r}.")


def lookup_external_forecast(
    data: pd.DataFrame,
    *,
    station: str,
    origin: datetime,
    horizon: int,
) -> pd.Series:
    """Look up exactly one external forecast available for an evaluation case."""
    origin_ts = pd.Timestamp(origin)
    matches = data[
        (data["station"] == station) & (data["forecast_origin"] == origin_ts) & (data["month_horizon"] == horizon)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one external forecast for "
            f"station={station!r}, origin={origin_ts.date()}, horizon={horizon}; found {len(matches)}."
        )
    row = matches.iloc[0]
    expected_target = origin_ts + pd.DateOffset(months=horizon)
    if not isclose((pd.Timestamp(row["target_month"]) - expected_target).total_seconds(), 0.0):
        raise ValueError("External forecast target date does not match origin + horizon.")
    return row


def deterministic_payload(value: float) -> ContinuousForecast:
    """Represent a point forecast in the shared continuous payload contract."""
    return ContinuousForecast(
        point_forecast=float(value),
        quantiles={quantile: float(value) for quantile in STANDARD_QUANTILES},
    )


class ExternalForecastPredictor(Predictor):
    """Adapter around the external XGBoost forecasts stored in the CSV."""

    def __init__(self, data: pd.DataFrame, *, forecast_scale: str) -> None:
        self._scale = normalize_forecast_scale(forecast_scale)
        self._forecast_column = forecast_column(self._scale)
        self._data = forecast_input_frame(data)

    @property
    def predictor_id(self) -> str:
        """Stable ID for persisted baseline predictions, unique per scale."""
        return f"external_xgboost_{self._scale}"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Return the external model's point forecast for the requested case."""
        if len(task.horizons) != 1:
            raise ValueError("AC One case specs must contain exactly one horizon.")
        if scale_for_task(task) != self._scale:
            raise ValueError(f"Predictor scale {self._scale!r} does not match task series {task.target_series_id!r}.")
        horizon = task.horizons[0]
        station = station_for_task(task, self._data)
        row = lookup_external_forecast(
            self._data,
            station=station,
            origin=context.as_of,
            horizon=horizon,
        )
        forecast_date = pd.Timestamp(row["target_month"]).to_pydatetime()
        point = float(row[self._forecast_column])
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
                as_of=context.as_of,
                forecast_date=forecast_date,
                payload=deterministic_payload(point),
                metadata={
                    "source": "external_xgboost_export",
                    "forecast_scale": self._scale,
                    "station": station,
                    "region": str(row["region"]),
                    "month_horizon": horizon,
                    "external_forecast": point,
                    "schedule_file_id": str(row["schd_file_id"]),
                    "model_version": str(row["model_version"]),
                },
            )
        ]


__all__ = [
    "ExternalForecastPredictor",
    "deterministic_payload",
    "lookup_external_forecast",
    "scale_for_task",
    "station_for_task",
]

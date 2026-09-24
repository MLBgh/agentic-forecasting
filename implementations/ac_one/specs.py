"""Experiment-manifest loading for AC One.

The CSV is a ragged panel when indexed by forecast origin: each origin has
forecasts at one, two, and/or three month leads, so target months differ by
horizon.  A single ``MultiTargetBacktestSpec`` cannot express different origin
lists per task, so the manifest below compiles to one standard ``BacktestSpec``
per station/horizon pair for a chosen forecast scale.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from ac_one.data import ForecastScale, normalize_forecast_scale, station_series_id
from aieng.forecasting.evaluation import BacktestSpec, ForecastingTask
from pydantic import BaseModel, Field, model_validator


DEFAULT_SPEC_PATH = Path(__file__).with_name("specs") / "ac_one_backtest.yaml"


class AcOneExperimentSpec(BaseModel):
    """Declarative filter that expands into station/horizon backtest specs."""

    spec_id: str = Field(min_length=1)
    description: str = ""
    stations: list[str] = Field(min_length=1)
    horizons: list[int] = Field(min_length=1)
    target_start: datetime
    target_end: datetime

    @model_validator(mode="after")
    def validate_contract(self) -> "AcOneExperimentSpec":
        """Validate date, station, and horizon uniqueness constraints."""
        if self.target_start > self.target_end:
            raise ValueError("target_start must be on or before target_end.")
        if len(self.stations) != len(set(self.stations)):
            raise ValueError("stations must not contain duplicates.")
        if len(self.horizons) != len(set(self.horizons)):
            raise ValueError("horizons must not contain duplicates.")
        if any(horizon < 1 for horizon in self.horizons):
            raise ValueError("horizons must contain positive integers.")
        return self


def load_experiment_spec(path: Path = DEFAULT_SPEC_PATH) -> AcOneExperimentSpec:
    """Load and validate an AC One YAML experiment manifest."""
    with path.open(encoding="utf-8") as handle:
        return AcOneExperimentSpec.model_validate(yaml.safe_load(handle))


def case_id(station: str, horizon: int) -> str:
    """Return the stable key used for one station/horizon evaluation case."""
    return f"{station.lower()}_{horizon}m"


def case_spec_id(experiment: AcOneExperimentSpec, station: str, horizon: int, scale: str) -> str:
    """Return a cache-safe spec ID for one compiled case and forecast scale."""
    return f"{experiment.spec_id}_{normalize_forecast_scale(scale)}_{case_id(station, horizon)}"


def build_backtest_specs(
    experiment: AcOneExperimentSpec,
    data: pd.DataFrame,
    *,
    forecast_scale: str,
) -> dict[str, BacktestSpec]:
    """Compile the manifest and normalized CSV into standard backtest specs."""
    scale: ForecastScale = normalize_forecast_scale(forecast_scale)
    available_stations = set(data["station"].astype(str))
    missing_stations = sorted(set(experiment.stations) - available_stations)
    if missing_stations:
        raise ValueError(f"Spec references stations absent from the dataset: {missing_stations}")

    available_horizons = set(data["month_horizon"].astype(int))
    missing_horizons = sorted(set(experiment.horizons) - available_horizons)
    if missing_horizons:
        raise ValueError(f"Spec references horizons absent from the dataset: {missing_horizons}")

    if "target_month" not in data.columns:
        raise ValueError("Normalized AC One data is missing target_month; load via load_forecast_data().")
    target_start = pd.Timestamp(experiment.target_start)
    target_end = pd.Timestamp(experiment.target_end)
    specs: dict[str, BacktestSpec] = {}

    for station in experiment.stations:
        for horizon in experiment.horizons:
            selected = data[
                (data["station"] == station)
                & (data["month_horizon"] == horizon)
                & (data["target_month"] >= target_start)
                & (data["target_month"] <= target_end)
            ].sort_values("forecast_origin")
            if selected.empty:
                raise ValueError(
                    f"No rows match station={station!r}, horizon={horizon}, "
                    f"target window [{target_start.date()}, {target_end.date()}]."
                )

            origins = [pd.Timestamp(value).to_pydatetime() for value in selected["forecast_origin"]]
            if len(origins) < 2:
                raise ValueError(
                    f"At least two rows are required for station={station!r}, horizon={horizon}; found {len(origins)}."
                )

            key = case_id(station, horizon)
            specs[key] = BacktestSpec(
                task=ForecastingTask(
                    task_id=f"fuel_consumption_{scale}_{key}",
                    target_series_id=station_series_id(station, scale),
                    horizons=[horizon],
                    frequency="MS",
                    description=(
                        f"Monthly airline fuel consumption for {station} on the {scale} scale, "
                        f"forecast {horizon} month(s) ahead."
                    ),
                ),
                start=min(origins),
                end=max(origins),
                origin_dates=origins,
                warmup=0,
                description=experiment.description,
            )

    return specs


__all__ = [
    "DEFAULT_SPEC_PATH",
    "AcOneExperimentSpec",
    "build_backtest_specs",
    "case_id",
    "case_spec_id",
    "load_experiment_spec",
]

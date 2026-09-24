"""Tests for AC One scale-isolated specs, baseline lookup, and prompt leakage."""

from __future__ import annotations

import pandas as pd
from ac_one.analyst_agent.agent import FuelForecastPromptBuilder
from ac_one.data import build_ac_one_service, forecast_column, load_forecast_data, target_month_for
from ac_one.predictors import ExternalForecastPredictor, station_for_task
from ac_one.specs import build_backtest_specs, case_spec_id, load_experiment_spec


def test_spec_ids_and_series_ids_are_scale_specific() -> None:
    """Minmax and indexed backtests must not share cache or target-series IDs."""
    data = load_forecast_data()
    experiment = load_experiment_spec()
    minmax_specs = build_backtest_specs(experiment, data, forecast_scale="minmax")
    indexed_specs = build_backtest_specs(experiment, data, forecast_scale="indexed")
    assert minmax_specs.keys() == indexed_specs.keys()
    case_name = next(iter(minmax_specs))
    minmax_spec = minmax_specs[case_name]
    indexed_spec = indexed_specs[case_name]
    assert minmax_spec.task.target_series_id != indexed_spec.task.target_series_id
    assert minmax_spec.task.target_series_id.endswith("_minmax")
    assert indexed_spec.task.target_series_id.endswith("_indexed")
    station = station_for_task(minmax_spec.task, data)
    horizon = minmax_spec.task.horizons[0]
    assert "minmax" in case_spec_id(experiment, station, horizon, "minmax")
    assert "indexed" in case_spec_id(experiment, station, horizon, "indexed")
    assert case_spec_id(experiment, station, horizon, "minmax") != case_spec_id(experiment, station, horizon, "indexed")


def test_external_predictor_reads_the_requested_scale() -> None:
    """Baseline points come from forecast_minmax or forecast_indexed, not a shared column."""
    data = load_forecast_data()
    experiment = load_experiment_spec()
    service = build_ac_one_service(data)
    for scale in ("minmax", "indexed"):
        specs = build_backtest_specs(experiment, data, forecast_scale=scale)
        spec = next(iter(specs.values()))
        origin = spec.origin_dates[0]
        predictor = ExternalForecastPredictor(data, forecast_scale=scale)
        assert predictor.predictor_id == f"external_xgboost_{scale}"
        predictions = predictor.predict(spec.task, service.context(origin))
        station = predictions[0].metadata["station"]
        row = data[
            (data["station"] == station)
            & (data["forecast_origin"] == origin)
            & (data["month_horizon"] == spec.task.horizons[0])
        ].iloc[0]
        assert predictions[0].payload.point_forecast == float(row[forecast_column(scale)])
        assert predictions[0].metadata["forecast_scale"] == scale


def test_forecast_date_is_the_target_month_at_every_lead() -> None:
    """Every lead must resolve to origin + (lead - 1), not to the run month or origin + lead."""
    data = load_forecast_data()
    experiment = load_experiment_spec()
    service = build_ac_one_service(data)
    predictor = ExternalForecastPredictor(data, forecast_scale="minmax")
    leads_exercised = set()
    for spec in build_backtest_specs(experiment, data, forecast_scale="minmax").values():
        horizon = spec.task.horizons[0]
        leads_exercised.add(horizon)
        for origin in spec.origin_dates:
            prediction = predictor.predict(spec.task, service.context(origin))[0]
            assert pd.Timestamp(prediction.forecast_date) == target_month_for(origin, horizon)
    # Guards against the suite silently collapsing to the lead where the run
    # month and the target month coincide.
    assert leads_exercised == {1, 2, 3}


def test_agent_prompt_excludes_actuals_and_uses_scale_forecast() -> None:
    """The agent prompt must not contain outcome columns and must echo the scale forecast."""
    data = load_forecast_data()
    experiment = load_experiment_spec()
    service = build_ac_one_service(data)
    for scale in ("minmax", "indexed"):
        specs = build_backtest_specs(experiment, data, forecast_scale=scale)
        spec = next(iter(specs.values()))
        origin = spec.origin_dates[0]
        prompt = FuelForecastPromptBuilder(data, forecast_scale=scale)(task=spec.task, context=service.context(origin))
        assert "actual_minmax" not in prompt
        assert "actual_indexed" not in prompt
        station = station_for_task(spec.task, data)
        match = data[
            (data["station"] == station)
            & (data["forecast_origin"] == origin)
            & (data["month_horizon"] == spec.task.horizons[0])
        ].iloc[0]
        assert str(float(match[forecast_column(scale)])) in prompt
        assert f'"forecast_scale": "{scale}"' in prompt

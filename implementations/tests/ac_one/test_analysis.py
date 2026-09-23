"""Tests for AC One MAE/MAPE scoring against the matching actual column."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from ac_one.analysis import metric_display_formats, metric_summary, paired_improvement, predictions_to_frame
from ac_one.data import actual_column, forecast_column, load_forecast_data
from ac_one.predictors import deterministic_payload, station_series_id
from ac_one.specs import build_backtest_specs, load_experiment_spec
from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.evaluation.prediction import Prediction


def _backtest_result(spec, prediction: Prediction) -> BacktestResult:
    return BacktestResult(
        spec=spec,
        predictor_id=prediction.predictor_id,
        predictions=[prediction],
        scores=[0.0],
        mean_score=0.0,
        ran_at=datetime(2026, 1, 1),
        skipped_origins=0,
    )


def _result_from_csv(scale: str) -> tuple[dict[str, dict[str, BacktestResult]], pd.DataFrame, float, float]:
    data = load_forecast_data()
    experiment = load_experiment_spec()
    specs = build_backtest_specs(experiment, data, forecast_scale=scale)
    case_name, spec = next(iter(specs.items()))
    origin = spec.origin_dates[0]
    station = [s for s in data["station"].unique() if station_series_id(str(s), scale) == spec.task.target_series_id][0]
    row = data[
        (data["station"] == station)
        & (data["forecast_origin"] == origin)
        & (data["month_horizon"] == spec.task.horizons[0])
    ].iloc[0]
    forecast = float(row[forecast_column(scale)])
    prediction = Prediction(
        predictor_id="external_xgboost",
        task_id=spec.task.task_id,
        issued_at=origin,
        as_of=origin,
        forecast_date=pd.Timestamp(row["horizon_date"]).to_pydatetime(),
        payload=deterministic_payload(forecast),
        metadata={"forecast_scale": scale, "station": station, "month_horizon": spec.task.horizons[0]},
    )
    result = _backtest_result(spec, prediction)
    return {"External XGBoost": {case_name: result}}, data, float(row[actual_column(scale)]), forecast


def test_minmax_mae_is_nonzero_and_not_integer_rounded_to_zero() -> None:
    """Minmax errors are small (~0.08) and must remain visible, not format as 0."""
    results, data, actual, forecast = _result_from_csv("minmax")
    scored = predictions_to_frame(results, data, forecast_scale="minmax")
    assert len(scored) == 1
    assert scored.loc[0, "actual"] == actual
    expected_mae = abs(forecast - actual)
    assert expected_mae > 0
    assert scored.loc[0, "absolute_error"] == expected_mae
    summary = metric_summary(scored)
    assert summary.loc[0, "mae"] == expected_mae
    mae_format = metric_display_formats("minmax")["mae"]
    formatted = mae_format.format(summary.loc[0, "mae"])
    assert formatted != "0"
    assert formatted != "0.0000"
    assert formatted == f"{expected_mae:.4f}"


def test_indexed_uses_matching_actual_not_minmax() -> None:
    """Indexed forecasts must not be scored against minmax actuals."""
    results, data, actual, forecast = _result_from_csv("indexed")
    scored = predictions_to_frame(results, data, forecast_scale="indexed")
    assert scored.loc[0, "actual"] == actual
    assert scored.loc[0, "forecast_scale"] == "indexed"
    assert scored.loc[0, "absolute_error"] == abs(forecast - actual)


def test_paired_improvement_is_zero_when_agent_copies_baseline() -> None:
    """Unchanged agent forecasts produce zero improvement, which is a real outcome."""
    results, data, _actual, _forecast = _result_from_csv("minmax")
    case_name, result = next(iter(results["External XGBoost"].items()))
    agent_pred = result.predictions[0].model_copy(
        update={
            "predictor_id": "agent",
            "metadata": {**result.predictions[0].metadata, "adjustment_pct": 0.0},
        }
    )
    scored = predictions_to_frame(
        {
            "External XGBoost": {case_name: result},
            "News-adjusted agent": {case_name: _backtest_result(result.spec, agent_pred)},
        },
        data,
        forecast_scale="minmax",
    )
    paired = paired_improvement(scored)
    assert paired.loc[0, "mean_mae_improvement"] == 0.0
    assert paired.loc[0, "mean_mape_improvement_pct_points"] == 0.0

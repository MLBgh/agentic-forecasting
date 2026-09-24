"""Tests for the AC One toolbelt fold, the Python adjustment cap, and prompt leakage."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd
import pytest
from ac_one.analyst_agent import (
    MAX_ABSOLUTE_ADJUSTMENT_PCT,
    FuelForecastPromptBuilder,
    build_fuel_adjustment_config,
    clip_adjustments,
    enabled_tool_labels,
    news_search,
)
from ac_one.data import build_ac_one_service, load_forecast_data
from ac_one.predictors import deterministic_payload
from ac_one.specs import build_backtest_specs, load_experiment_spec
from aieng.forecasting.evaluation import Prediction


def test_default_toolbelt_is_news_on_and_empty_toolbelt_is_a_control() -> None:
    """No ``tools`` argument keeps search on; ``tools=()`` removes search, skill, and supplement."""
    default = build_fuel_adjustment_config(forecast_scale="indexed")
    explicit = build_fuel_adjustment_config(forecast_scale="indexed", tools=[news_search()])
    control = build_fuel_adjustment_config(forecast_scale="indexed", tools=())

    assert default.context_retrieval.enabled
    assert default.instruction == explicit.instruction
    assert [p.name for p in default.skills_dirs] == ["news-adjustment"]
    assert enabled_tool_labels(default) == ["news_search"]

    assert not control.context_retrieval.enabled
    assert list(control.skills_dirs) == []
    assert "search_web" not in control.instruction
    assert enabled_tool_labels(control) == []

    assert default.temperature == 0.0 == control.temperature
    assert default.name == control.name == "ac_one_fuel_news_adjuster_indexed"


def _prediction(baseline: float, adjusted: float) -> Prediction:
    return Prediction(
        predictor_id="agent",
        task_id="t",
        issued_at=datetime(2026, 1, 1),
        as_of=datetime(2026, 1, 1),
        forecast_date=datetime(2026, 2, 1),
        payload=deterministic_payload(adjusted),
        metadata={"baseline_forecast": baseline, "adjustment_pct": (adjusted / baseline - 1.0) * 100.0},
    )


@pytest.mark.parametrize(
    ("adjusted", "cap_pct", "expected_point", "expected_pct", "capped"),
    [
        (135.0, 20.0, 120.0, 20.0, True),
        (70.0, 20.0, 80.0, -20.0, True),
        (110.0, 20.0, 110.0, 10.0, False),
        (110.0, 5.0, 105.0, 5.0, True),
        (100.0, 0.0, 100.0, 0.0, False),
    ],
)
def test_clip_adjustments_bounds_point_and_recomputes_pct(
    adjusted: float, cap_pct: float, expected_point: float, expected_pct: float, capped: bool
) -> None:
    """The payload, every quantile, and adjustment_pct all reflect the clipped point."""
    prediction = _prediction(100.0, adjusted)
    clip_adjustments([prediction], cap_pct=cap_pct)
    payload: Any = prediction.payload
    assert payload.point_forecast == pytest.approx(expected_point)
    assert all(q == pytest.approx(expected_point) for q in payload.quantiles.values())
    assert prediction.metadata["adjustment_pct"] == pytest.approx(expected_pct)
    assert prediction.metadata["raw_adjustment_pct"] == pytest.approx((adjusted / 100.0 - 1.0) * 100.0)
    assert prediction.metadata["cap_applied"] is capped
    assert prediction.metadata["cap_pct"] == cap_pct
    assert prediction.metadata["baseline_forecast"] == 100.0


def _keys(node: Any) -> set[str]:
    if isinstance(node, dict):
        return set(node) | {k for v in node.values() for k in _keys(v)}
    if isinstance(node, list):
        return {k for v in node for k in _keys(v)}
    return set()


def test_prompt_payload_has_no_actual_keys_and_carries_the_cap() -> None:
    """Every scale's prompt JSON is outcome-free and states the configured cap."""
    data = load_forecast_data()
    experiment = load_experiment_spec()
    service = build_ac_one_service(data)
    for scale in ("minmax", "indexed"):
        spec = next(iter(build_backtest_specs(experiment, data, forecast_scale=scale).values()))
        origin = spec.origin_dates[0]
        for cap in (MAX_ABSOLUTE_ADJUSTMENT_PCT, 7.5):
            builder = FuelForecastPromptBuilder(data, forecast_scale=scale, cap_pct=cap)
            payload = json.loads(builder(task=spec.task, context=service.context(origin)))
            assert not {k for k in _keys(payload) if "actual" in k and k != "actual_is_unavailable"}
            assert payload["constraints"]["max_absolute_adjustment_pct"] == cap
            assert payload["as_of"] == str(pd.Timestamp(origin).date())
            assert payload["target_month"] != payload["as_of"]
            assert payload["station"] in {"STATION_A", "STATION_B"}

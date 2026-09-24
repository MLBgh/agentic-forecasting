"""Tests for the AC One trace stamp the trace evaluator reads back (no network)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from ac_one.data import load_forecast_data
from ac_one.predictors import deterministic_payload
from ac_one.trace_eval import evaluate_trace_forecasts
from ac_one.trace_stamp import stamp_continuous_forecasts
from aieng.forecasting.evaluation import Prediction
from aieng.forecasting.evaluation.langfuse_traces import read_forecasts_from_trace


class _FakeObservation:
    def __init__(self) -> None:
        self.name = "forecast"
        self.output: Any = None

    def update(self, **kwargs: Any) -> None:
        self.output = kwargs.get("output")

    def __enter__(self) -> "_FakeObservation":
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False


class _FakeTrace:
    def __init__(self, observation: _FakeObservation) -> None:
        self.observations = [observation]
        self.output: Any = None


class _FakeClient:
    """Captures the stamped observation and serves it back as a fetched trace."""

    def __init__(self) -> None:
        self.observation = _FakeObservation()
        self.scores: list[dict[str, Any]] = []

    def start_as_current_observation(self, **_kwargs: Any) -> _FakeObservation:
        return self.observation

    def create_score(self, **kwargs: Any) -> None:
        self.scores.append(kwargs)

    def get_trace_url(self, *, trace_id: str | None = None) -> str:
        return f"https://langfuse.example/traces/{trace_id}"

    def flush(self) -> None:
        pass

    def trace(self) -> _FakeTrace:
        return _FakeTrace(self.observation)


def _enriched_prediction(row: pd.Series, *, rationale: str) -> Prediction:
    """Build a prediction shaped the way ``FuelAdjustmentPredictor`` enriches one."""
    return Prediction(
        predictor_id="agent_predictor_ac_one_fuel_news_adjuster_minmax",
        task_id="fuel_consumption_minmax_case",
        issued_at=datetime(2026, 1, 1),
        as_of=pd.Timestamp(row["forecast_origin"]).to_pydatetime(),
        forecast_date=pd.Timestamp(row["target_month"]).to_pydatetime(),
        payload=deterministic_payload(float(row["forecast_minmax"])),
        metadata={
            "rationale": rationale,
            "evidence_summary": "Insufficient cutoff-safe news.",
            "source_urls": ["https://example.com"],
            "baseline_forecast": float(row["forecast_minmax"]),
            "adjustment_pct": 0.0,
            "station": str(row["station"]),
            "forecast_scale": "minmax",
            "month_horizon": int(row["month_horizon"]),
        },
    )


def test_stamped_point_forecast_scores_through_the_trace_evaluator() -> None:
    """The stamp must carry everything ``evaluate_trace_forecasts`` joins actuals on.

    The shared library stamps categorical forecasts only, so this round trip —
    stamp, read back with the library reader, score — is the contract that keeps
    AC One's overlay scoreable without touching the core module.
    """
    data = load_forecast_data()
    row = data.iloc[0]
    client = _FakeClient()

    assert stamp_continuous_forecasts([_enriched_prediction(row, rationale="Retain baseline.")], client=client)

    stamped = read_forecasts_from_trace(client.trace())
    assert stamped[0]["payload_type"] == "continuous"
    assert stamped[0]["point_forecast"] == float(row["forecast_minmax"])
    assert stamped[0]["baseline_forecast"] == float(row["forecast_minmax"])

    scored = evaluate_trace_forecasts(
        ["t-1"],
        data,
        push_to_langfuse=False,
        client=client,
        fetch=lambda _trace_id, client=None, **_kwargs: client.trace(),
    )

    assert len(scored) == 1
    assert scored.loc[0, "station"] == str(row["station"])
    assert scored.loc[0, "forecast_scale"] == "minmax"
    assert scored.loc[0, "horizon"] == int(row["month_horizon"])
    assert scored.loc[0, "actual"] == float(row["actual_minmax"])


def test_forecast_without_a_rationale_is_not_stamped() -> None:
    """A rationale-free forecast gives the trace evaluator nothing to score."""
    row = load_forecast_data().iloc[0]
    client = _FakeClient()

    assert stamp_continuous_forecasts([_enriched_prediction(row, rationale="")], client=client) is False
    assert client.observation.output is None

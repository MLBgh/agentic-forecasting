"""Tests for AC One Langfuse trace scoring (no network; fake client)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from ac_one.data import load_forecast_data
from ac_one.specs import case_id
from ac_one.trace_eval import RationaleQualityVerdict, evaluate_trace_forecasts


class _FakeTrace:
    def __init__(self, forecasts: list[dict[str, Any]]) -> None:
        self.output = {"forecasts": forecasts}


class _FakeClient:
    def __init__(self, traces: dict[str, _FakeTrace]) -> None:
        self._traces = traces
        self.scores: list[dict[str, Any]] = []
        self.flushed = 0
        client = self

        class _TraceApi:
            def get(self, trace_id: str, **_kwargs: Any) -> _FakeTrace:
                return client._traces[trace_id]

        self.api = type("_Api", (), {"trace": _TraceApi()})()

    def create_score(self, **kwargs: Any) -> None:
        self.scores.append(kwargs)

    def get_trace_url(self, *, trace_id: str | None = None) -> str:
        return f"https://langfuse.example/project/air-canada-1/traces/{trace_id}"

    def flush(self) -> None:
        self.flushed += 1


def test_evaluate_trace_forecasts_pushes_deterministic_and_judge_scores() -> None:
    data = load_forecast_data()
    row = data.iloc[0]
    scale = "minmax"
    target = pd.Timestamp(row["target_month"])
    forecast = {
        "payload_type": "continuous",
        "predictor_id": "agent",
        "forecast_date": target.isoformat(),
        "as_of": pd.Timestamp(row["horizon_date"]).isoformat(),
        "station": str(row["station"]),
        "forecast_scale": scale,
        "month_horizon": int(row["month_horizon"]),
        "point_forecast": float(row["forecast_minmax"]),
        "baseline_forecast": float(row["forecast_minmax"]),
        "adjustment_pct": 0.0,
        "rationale": "No incremental evidence; retain baseline.",
        "evidence_summary": "Insufficient cutoff-safe news.",
        "source_urls": ["https://example.com"],
    }
    client = _FakeClient({"t-1": _FakeTrace([forecast])})

    def _judge(**_kwargs: Any) -> RationaleQualityVerdict:
        return RationaleQualityVerdict(
            quality_score=0.7,
            adjustment_justified=True,
            cutoff_safe=True,
            justification="Zero adjustment is conservative given weak evidence.",
        )

    scored = evaluate_trace_forecasts(
        ["t-1"],
        data,
        push_to_langfuse=True,
        run_judge=True,
        judge=_judge,
        client=client,
        fetch=lambda trace_id, client=None, **_k: client._traces[trace_id],
    )

    assert len(scored) == 1
    assert scored.loc[0, "forecast_scale"] == "minmax"
    assert scored.loc[0, "langfuse_scored"]
    names = {item["name"] for item in client.scores}
    assert {"absolute_error", "ape_pct", "mae_improvement", "mape_improvement_pct_points", "adjustment_pct"} <= names
    assert "rationale_quality" in names
    assert client.flushed == 1
    abs_error = next(item for item in client.scores if item["name"] == "absolute_error")
    assert abs_error["value"] == abs(float(row["forecast_minmax"]) - float(row["actual_minmax"]))
    assert abs_error["value"] > 0


def test_identity_is_recovered_from_task_id_when_the_stamp_omits_it() -> None:
    """Traces stamped before the predictor attached identity are still scoreable.

    Those payloads carry empty ``station``/``forecast_scale`` and a null
    ``month_horizon``, leaving ``task_id`` as the only identity. Without
    recovery the trace is silently skipped and the run scores nothing.
    """
    data = load_forecast_data()
    row = data.iloc[0]
    station = str(row["station"])
    horizon = int(row["month_horizon"])
    target = pd.Timestamp(row["target_month"])
    forecast = {
        "payload_type": "continuous",
        "predictor_id": "agent",
        "task_id": f"fuel_consumption_minmax_{case_id(station, horizon)}",
        "forecast_date": target.isoformat(),
        "as_of": pd.Timestamp(row["horizon_date"]).isoformat(),
        "station": "",
        "forecast_scale": "",
        "month_horizon": None,
        "point_forecast": float(row["forecast_minmax"]),
        "baseline_forecast": float(row["forecast_minmax"]),
        "adjustment_pct": 0.0,
        "rationale": "No incremental evidence; retain baseline.",
        "evidence_summary": "Insufficient cutoff-safe news.",
        "source_urls": [],
    }
    client = _FakeClient({"t-1": _FakeTrace([forecast])})

    scored = evaluate_trace_forecasts(
        ["t-1"],
        data,
        push_to_langfuse=False,
        client=client,
        fetch=lambda trace_id, client=None, **_k: client._traces[trace_id],
    )

    assert len(scored) == 1
    assert scored.loc[0, "station"] == station
    assert scored.loc[0, "forecast_scale"] == "minmax"
    assert scored.loc[0, "horizon"] == horizon
    assert scored.loc[0, "actual"] == float(row["actual_minmax"])

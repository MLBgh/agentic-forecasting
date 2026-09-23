"""Stamp AC One point forecasts onto their Langfuse traces.

The shared :func:`aieng.forecasting.evaluation.langfuse_traces.stamp_forecast_on_trace`
projects categorical distributions, so the news-adjustment overlay writes its own
continuous payload. It reuses the library's observation name and output key, which
keeps :func:`~aieng.forecasting.evaluation.langfuse_traces.read_forecasts_from_trace`
— and therefore :mod:`ac_one.trace_eval` — as the reader on the other side.

The stamp is the canonical record the trace evaluator scores from, so everything it
needs (station, forecast scale, horizon, baseline, adjustment) must be on the
prediction metadata before stamping. Best-effort throughout: a missing or
misconfigured Langfuse degrades to a no-op rather than failing a backtest.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from aieng.forecasting.evaluation import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.langfuse_traces import (
    FORECAST_OBSERVATION_NAME,
    FORECAST_TRACE_OUTPUT_KEY,
)


logger = logging.getLogger(__name__)


def _isoformat(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def continuous_forecast_to_dict(pred: Prediction) -> dict[str, Any] | None:
    """Project one rationale-bearing point forecast to a trace-output dict.

    Returns ``None`` for predictions the trace evaluator cannot score — a
    non-continuous payload, or one with no stated rationale — so they are not
    stamped.
    """
    if not isinstance(pred.payload, ContinuousForecast):
        return None
    metadata = pred.metadata or {}
    rationale = str(metadata.get("rationale", "") or "").strip()
    if not rationale:
        return None
    return {
        "payload_type": "continuous",
        "predictor_id": pred.predictor_id,
        "task_id": pred.task_id,
        "forecast_date": _isoformat(pred.forecast_date),
        "as_of": _isoformat(pred.as_of),
        "rationale": rationale,
        "station": str(metadata.get("station", "") or ""),
        "forecast_scale": str(metadata.get("forecast_scale", "") or ""),
        "month_horizon": metadata.get("month_horizon"),
        "evidence_summary": str(metadata.get("evidence_summary", "") or ""),
        "source_urls": list(metadata.get("source_urls", []) or []),
        "point_forecast": float(pred.payload.point_forecast),
        "baseline_forecast": metadata.get("baseline_forecast"),
        "adjustment_pct": metadata.get("adjustment_pct"),
    }


def stamp_continuous_forecasts(
    predictions: Sequence[Prediction],
    *,
    trace_id: str | None = None,
    client: Any | None = None,
) -> bool:
    """Write the adjusted forecast(s) onto a ``forecast`` observation in the trace.

    Parameters
    ----------
    predictions : sequence of Prediction
        The predictions to stamp (filtered to rationale-bearing point forecasts).
    trace_id : str or None
        When given, the observation is attached to that trace **post-hoc**. The
        agent runs on a worker event loop whose trace context is not active in the
        caller, so this is the usual path; ``None`` uses the active context.
    client : Langfuse client, optional
        Defaults to the process-wide client.

    Returns ``True`` when something was stamped, ``False`` on no-op (nothing to
    stamp, or Langfuse unavailable). Never raises.
    """
    forecasts = [d for d in (continuous_forecast_to_dict(p) for p in predictions) if d is not None]
    if not forecasts:
        return False
    try:
        if client is None:
            from langfuse import get_client  # noqa: PLC0415

            client = get_client()
        kwargs: dict[str, Any] = {"name": FORECAST_OBSERVATION_NAME, "as_type": "span"}
        if trace_id is not None:
            from langfuse.types import TraceContext  # noqa: PLC0415

            kwargs["trace_context"] = TraceContext(trace_id=trace_id)
        with client.start_as_current_observation(**kwargs) as observation:
            observation.update(output={FORECAST_TRACE_OUTPUT_KEY: forecasts})
        return True
    except Exception:  # pragma: no cover - guarded no-op when tracing is unavailable
        logger.debug("Could not stamp AC One forecast onto Langfuse trace.", exc_info=True)
        return False


__all__ = [
    "continuous_forecast_to_dict",
    "stamp_continuous_forecasts",
]

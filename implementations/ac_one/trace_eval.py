"""Langfuse trace evaluation for AC One news-adjusted forecasts.

Reads stamped continuous forecasts from Langfuse (project ``air-canada-1``),
joins the matching evaluation-only actual after prediction, and pushes
deterministic error scores. An optional LLM judge scores rationale quality
on the same traces.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

import pandas as pd
from ac_one.analyst_agent.agent import LANGFUSE_PROJECT_NAME
from ac_one.data import FORECAST_SCALES, actual_column, normalize_forecast_scale
from ac_one.specs import case_id
from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.evaluation.langfuse_traces import (
    fetch_trace_with_wait,
    flush_scores,
    push_trace_score,
    read_forecasts_from_trace,
)
from aieng.forecasting.langfuse_tracing import init_langfuse_tracing
from aieng.forecasting.models import ADVANCED_MODEL
from pydantic import BaseModel, Field


class LangfuseConnection:
    """Result of authenticating to the configured Langfuse project."""

    def __init__(self, *, ok: bool, host: str, message: str, client: Any | None = None) -> None:
        self.ok = ok
        self.host = host
        self.message = message
        self.client = client


def connect_langfuse() -> LangfuseConnection:
    """Initialize Langfuse from ``LANGFUSE_*`` env vars and verify authentication.

    Keys select the destination project. They must belong to ``air-canada-1``.
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    host = (os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL") or "").strip()
    display_host = host or "https://cloud.langfuse.com"
    if not public_key or not secret_key:
        return LangfuseConnection(
            ok=False,
            host=display_host,
            message=(
                "Langfuse credentials are missing. Set LANGFUSE_PUBLIC_KEY, "
                "LANGFUSE_SECRET_KEY, and LANGFUSE_HOST for project "
                f"{LANGFUSE_PROJECT_NAME!r}, then reload the kernel."
            ),
        )
    try:
        from langfuse import get_client  # noqa: PLC0415
    except ImportError:
        return LangfuseConnection(
            ok=False,
            host=display_host,
            message="langfuse is not installed. Run `uv sync` with the agentic extra.",
        )
    init_langfuse_tracing()
    client = get_client()
    try:
        authenticated = bool(client.auth_check())
    except Exception as exc:  # noqa: BLE001
        return LangfuseConnection(
            ok=False,
            host=display_host,
            client=client,
            message=f"Langfuse auth failed for {LANGFUSE_PROJECT_NAME!r} at {display_host}: {exc}",
        )
    if not authenticated:
        return LangfuseConnection(
            ok=False,
            host=display_host,
            client=client,
            message=(
                f"Langfuse auth_check() returned False. Re-check keys for project "
                f"{LANGFUSE_PROJECT_NAME!r} and LANGFUSE_HOST={display_host}."
            ),
        )
    return LangfuseConnection(
        ok=True,
        host=display_host,
        client=client,
        message=f"Authenticated to Langfuse project {LANGFUSE_PROJECT_NAME!r} at {display_host}.",
    )


class RationaleQualityVerdict(BaseModel):
    """LLM-as-judge assessment of one news-adjustment rationale."""

    quality_score: float = Field(ge=0.0, le=1.0)
    adjustment_justified: bool
    cutoff_safe: bool
    justification: str = ""


_QUALITY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "quality_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "adjustment_justified": {"type": "boolean"},
        "cutoff_safe": {"type": "boolean"},
        "justification": {"type": "string"},
    },
    "required": ["quality_score", "adjustment_justified", "cutoff_safe", "justification"],
    "additionalProperties": False,
}

_JUDGE_SYSTEM_PROMPT = (
    "You evaluate airline fuel-consumption forecast adjustments. Judge the "
    "forecaster's rationale and evidence, not numerical accuracy.\n"
    "\n"
    "Rules:\n"
    "- Return ONLY a JSON object matching the provided schema.\n"
    "- quality_score in [0, 1] rates process quality: cutoff-safe sources, "
    "relevance to physical fuel consumption (not price alone), and whether the "
    "stated adjustment is conservative and supported.\n"
    "- adjustment_justified is true only if the evidence plausibly supports the "
    "signed adjustment (including a justified zero adjustment).\n"
    "- cutoff_safe is true only if the rationale does not rely on information "
    "after the as_of date.\n"
    "- justification is 2-3 sentences."
)


def _build_judge_user_prompt(
    *,
    station: str,
    forecast_scale: str,
    as_of: str,
    target_month: str,
    baseline_forecast: float | None,
    adjusted_forecast: float,
    adjustment_pct: float | None,
    rationale: str,
    evidence_summary: str,
    source_urls: list[str],
) -> str:
    sources = "; ".join(source_urls) if source_urls else "(none provided)"
    baseline = "unknown" if baseline_forecast is None else f"{baseline_forecast}"
    adj = "unknown" if adjustment_pct is None else f"{adjustment_pct}"
    return (
        f"Station: {station} (anonymized)\n"
        f"Forecast scale: {forecast_scale}\n"
        f"as_of cutoff: {as_of}\n"
        f"Target month: {target_month}\n"
        f"Baseline forecast: {baseline}\n"
        f"Adjusted forecast: {adjusted_forecast}\n"
        f"Adjustment pct: {adj}\n"
        "\n"
        f"Rationale:\n{rationale}\n"
        "\n"
        f"Evidence summary:\n{evidence_summary or '(none)'}\n"
        f"Source URLs: {sources}\n"
    )


def judge_rationale_quality(
    *,
    station: str,
    forecast_scale: str,
    as_of: str,
    target_month: str,
    baseline_forecast: float | None,
    adjusted_forecast: float,
    adjustment_pct: float | None,
    rationale: str,
    evidence_summary: str,
    source_urls: list[str],
    model: str = ADVANCED_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout_s: float = 120.0,
) -> RationaleQualityVerdict:
    """Run one LLM-as-judge call assessing news-adjustment rationale quality."""
    from aieng.forecasting.methods.llm_processes._client import (  # noqa: PLC0415
        make_json_schema_response_format,
        run_async,
        sample_n_async,
    )

    base_messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_judge_user_prompt(
                station=station,
                forecast_scale=forecast_scale,
                as_of=as_of,
                target_month=target_month,
                baseline_forecast=baseline_forecast,
                adjusted_forecast=adjusted_forecast,
                adjustment_pct=adjustment_pct,
                rationale=rationale,
                evidence_summary=evidence_summary,
                source_urls=source_urls,
            ),
        },
    ]
    response_format = make_json_schema_response_format("RationaleQuality", _QUALITY_JSON_SCHEMA)
    parsed, _cost, _in, _out, _fails = run_async(
        sample_n_async(
            schema_cls=RationaleQualityVerdict,
            model=model,
            base_messages=base_messages,
            response_format=response_format,
            n_samples=1,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            api_base=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
    )
    if not parsed:
        raise RuntimeError("Rationale-quality judge returned no schema-valid verdict.")
    return parsed[0]


def resolve_trace_url(trace_id: str, *, client: Any | None = None) -> str | None:
    """Return the Langfuse UI URL for ``trace_id``, or ``None`` if unavailable."""
    try:
        if client is None:
            from langfuse import get_client  # noqa: PLC0415

            client = get_client()
        return client.get_trace_url(trace_id=trace_id)
    except Exception:
        return None


def trace_ids_from_results(results: dict[str, BacktestResult]) -> list[str]:
    """Collect distinct Langfuse trace ids referenced by cached predictions."""
    seen: dict[str, None] = {}
    for result in results.values():
        for pred in result.predictions:
            trace_id = (pred.metadata or {}).get("langfuse_trace_id")
            if isinstance(trace_id, str) and trace_id:
                seen.setdefault(trace_id, None)
    return list(seen)


def _lookup_actual(data: pd.DataFrame, *, station: str, target: pd.Timestamp, scale: str) -> float:
    column = actual_column(scale)
    matches = data[(data["station"] == station) & (data["horizon_date"] == target)]
    if matches.empty:
        raise ValueError(f"No actual for station={station!r}, target={target.date()}, scale={scale}.")
    actual = float(matches.iloc[0][column])
    if actual == 0:
        raise ValueError(f"MAPE is undefined for zero actual at station={station!r}, target={target.date()}.")
    return actual


def task_identity_lookup(data: pd.DataFrame) -> dict[str, tuple[str, str, int]]:
    """Map each compiled task ID to its ``(station, scale, horizon)``.

    Traces stamped before the predictor attached station/scale/horizon metadata
    carry those fields as empty strings, leaving ``task_id`` as the only
    identity on the payload. Rebuilding the IDs ``ac_one.specs`` generates lets
    those traces still be scored without re-running the agent.
    """
    lookup: dict[str, tuple[str, str, int]] = {}
    horizons = sorted({int(horizon) for horizon in data["month_horizon"]})
    for station in sorted({str(value) for value in data["station"]}):
        for scale in FORECAST_SCALES:
            for horizon in horizons:
                lookup[f"fuel_consumption_{scale}_{case_id(station, horizon)}"] = (station, scale, horizon)
    return lookup


def _parse_continuous_forecast(
    forecast: dict[str, Any],
    *,
    identity: dict[str, tuple[str, str, int]] | None = None,
) -> dict[str, Any] | None:
    if forecast.get("payload_type") not in (None, "continuous"):
        return None
    point = forecast.get("point_forecast")
    rationale = str(forecast.get("rationale", "") or "").strip()
    scale = str(forecast.get("forecast_scale") or "")
    station = str(forecast.get("station") or "")
    month_horizon = forecast.get("month_horizon")
    if not (scale and station and month_horizon is not None) and identity:
        recovered = identity.get(str(forecast.get("task_id", "") or ""))
        if recovered is not None:
            station = station or recovered[0]
            scale = scale or recovered[1]
            month_horizon = recovered[2] if month_horizon is None else month_horizon
    if point is None or not rationale or not scale or not station:
        return None
    return {
        "point": float(point),
        "rationale": rationale,
        "scale": scale,
        "station": station,
        "target": pd.Timestamp(forecast["forecast_date"]),
        "baseline": forecast.get("baseline_forecast"),
        "adjustment_pct": forecast.get("adjustment_pct"),
        "as_of": str(forecast.get("as_of") or ""),
        "evidence_summary": str(forecast.get("evidence_summary", "") or ""),
        "source_urls": list(forecast.get("source_urls", []) or []),
        "predictor_id": str(forecast.get("predictor_id", "") or ""),
        "month_horizon": month_horizon,
    }


def _push_deterministic_scores(
    trace_id: str,
    *,
    client: Any | None,
    metadata: dict[str, str],
    absolute_error: float,
    ape_pct: float,
    mae_improvement: float | None,
    mape_improvement: float | None,
    adjustment_pct: Any,
) -> bool:
    scored = push_trace_score(trace_id, "absolute_error", float(absolute_error), client=client, metadata=metadata)
    push_trace_score(trace_id, "ape_pct", float(ape_pct), client=client, metadata=metadata)
    if mae_improvement is not None:
        push_trace_score(trace_id, "mae_improvement", float(mae_improvement), client=client, metadata=metadata)
    if mape_improvement is not None:
        push_trace_score(
            trace_id, "mape_improvement_pct_points", float(mape_improvement), client=client, metadata=metadata
        )
    if adjustment_pct is not None:
        push_trace_score(trace_id, "adjustment_pct", float(adjustment_pct), client=client, metadata=metadata)
    return scored


def _push_judge_scores(
    trace_id: str,
    verdict: RationaleQualityVerdict,
    *,
    client: Any | None,
    scale: str,
    station: str,
) -> None:
    push_trace_score(
        trace_id,
        "rationale_quality",
        float(verdict.quality_score),
        client=client,
        comment=verdict.justification,
        metadata={"forecast_scale": scale, "station": station},
    )
    push_trace_score(trace_id, "adjustment_justified", bool(verdict.adjustment_justified), client=client)
    push_trace_score(trace_id, "cutoff_safe", bool(verdict.cutoff_safe), client=client)


def evaluate_trace_forecasts(
    trace_ids: Sequence[str],
    data: pd.DataFrame,
    *,
    push_to_langfuse: bool = True,
    run_judge: bool = False,
    model: str = ADVANCED_MODEL,
    judge: Callable[..., RationaleQualityVerdict] = judge_rationale_quality,
    client: Any | None = None,
    fetch: Callable[..., Any] = fetch_trace_with_wait,
    max_wait_s: float = 5.0,
) -> pd.DataFrame:
    """Score stamped AC One forecasts from Langfuse traces.

    Deterministic scores (absolute error, APE, baseline improvements, adjustment)
    are computed against the matching ``actual_*`` column. The LLM judge is
    optional and is skipped unless ``run_judge`` is true.

    ``max_wait_s`` bounds the per-trace readiness poll. The default suits
    already-ingested traces; raise it only when scoring a run that just
    finished, since every unresolvable trace ID costs that full budget.
    """
    identity = task_identity_lookup(data)
    rows: list[dict[str, object]] = []
    pushed_any = False
    for trace_id in trace_ids:
        trace = fetch(trace_id, client=client, max_wait_s=max_wait_s)
        if trace is None:
            continue
        for forecast in read_forecasts_from_trace(trace):
            parsed = _parse_continuous_forecast(forecast, identity=identity)
            if parsed is None:
                continue
            actual = _lookup_actual(data, station=parsed["station"], target=parsed["target"], scale=parsed["scale"])
            adjusted = parsed["point"]
            absolute_error = abs(adjusted - actual)
            ape_pct = absolute_error / abs(actual) * 100.0
            baseline = parsed["baseline"]
            baseline_error = abs(float(baseline) - actual) if baseline is not None else None
            mae_improvement = (baseline_error - absolute_error) if baseline_error is not None else None
            baseline_ape = (baseline_error / abs(actual) * 100.0) if baseline_error is not None else None
            mape_improvement = (baseline_ape - ape_pct) if baseline_ape is not None else None
            adjustment_pct = parsed["adjustment_pct"]

            langfuse_scored = False
            if push_to_langfuse:
                langfuse_scored = _push_deterministic_scores(
                    trace_id,
                    client=client,
                    metadata={
                        "predictor_id": parsed["predictor_id"],
                        "forecast_scale": parsed["scale"],
                        "station": parsed["station"],
                        "langfuse_project": LANGFUSE_PROJECT_NAME,
                    },
                    absolute_error=absolute_error,
                    ape_pct=ape_pct,
                    mae_improvement=mae_improvement,
                    mape_improvement=mape_improvement,
                    adjustment_pct=adjustment_pct,
                )
                pushed_any = pushed_any or langfuse_scored

            verdict: RationaleQualityVerdict | None = None
            if run_judge:
                verdict = judge(
                    station=parsed["station"],
                    forecast_scale=parsed["scale"],
                    as_of=parsed["as_of"],
                    target_month=parsed["target"].date().isoformat(),
                    baseline_forecast=float(baseline) if baseline is not None else None,
                    adjusted_forecast=adjusted,
                    adjustment_pct=float(adjustment_pct) if adjustment_pct is not None else None,
                    rationale=parsed["rationale"],
                    evidence_summary=parsed["evidence_summary"],
                    source_urls=parsed["source_urls"],
                    model=model,
                )
                if push_to_langfuse:
                    _push_judge_scores(
                        trace_id, verdict, client=client, scale=parsed["scale"], station=parsed["station"]
                    )
                    pushed_any = True

            rows.append(
                {
                    "forecast_scale": normalize_forecast_scale(parsed["scale"]),
                    "station": parsed["station"],
                    "target_month": parsed["target"],
                    "horizon": parsed["month_horizon"],
                    "forecast": adjusted,
                    "actual": actual,
                    "absolute_error": absolute_error,
                    "ape_pct": ape_pct,
                    "baseline_forecast": baseline,
                    "mae_improvement": mae_improvement,
                    "mape_improvement_pct_points": mape_improvement,
                    "adjustment_pct": adjustment_pct,
                    "rationale_quality": None if verdict is None else verdict.quality_score,
                    "adjustment_justified": None if verdict is None else verdict.adjustment_justified,
                    "cutoff_safe": None if verdict is None else verdict.cutoff_safe,
                    "judge_justification": None if verdict is None else verdict.justification,
                    "langfuse_trace_id": trace_id,
                    "langfuse_trace_url": resolve_trace_url(trace_id, client=client),
                    "langfuse_scored": langfuse_scored,
                }
            )

    if pushed_any:
        flush_scores(client)
    return pd.DataFrame(rows)


__all__ = [
    "LANGFUSE_PROJECT_NAME",
    "LangfuseConnection",
    "RationaleQualityVerdict",
    "connect_langfuse",
    "evaluate_trace_forecasts",
    "judge_rationale_quality",
    "resolve_trace_url",
    "task_identity_lookup",
    "trace_ids_from_results",
]

"""Cutoff-aware news agent that adjusts external fuel-consumption forecasts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from math import copysign, isclose, isfinite
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from ac_one.analyst_agent.tools import ToolSpec, news_search
from ac_one.data import (
    ForecastScale,
    forecast_column,
    forecast_input_frame,
    normalize_forecast_scale,
)
from ac_one.predictors import (
    deterministic_payload,
    lookup_external_forecast,
    scale_for_task,
    station_for_task,
)
from ac_one.trace_stamp import stamp_continuous_forecasts
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation import ForecastingTask, Prediction
from aieng.forecasting.methods.agentic import AgentPredictor, build_adk_agent
from aieng.forecasting.methods.agentic.adk_runner import AdkTextRunner, AdkTextRunnerConfig
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.methods.agentic.outputs import AgentForecastOutput
from aieng.forecasting.models import LITE_MODEL
from pydantic import BaseModel, Field, field_validator, model_validator


MAX_ABSOLUTE_ADJUSTMENT_PCT = 20.0
LANGFUSE_PROJECT_NAME = "air-canada-1"

_ANALYST_INSTRUCTION = """\
## Role

You are an airline fuel-consumption forecasting analyst. An external XGBoost
model has already incorporated the organization's structured operational
features. Your job is narrow: decide whether exceptional external conditions,
known on or before the payload `as_of` date, justify a conservative adjustment
to that model's point forecast.

## Constraints

1. Information cutoff: use nothing that became known after `as_of`.
2. Default is zero: keep `adjusted_forecast` equal to `baseline_forecast` when
   evidence is absent, weak, stale, conflicting, or already part of normal
   operations. If you have no research tool, return a zero adjustment.
3. Cap: `|adjustment_pct|` may not exceed the payload
   `constraints.max_absolute_adjustment_pct`. Any larger value is clipped to
   the cap after the fact.
4. The station identifier is anonymized. Never infer or name an airport, city,
   carrier, route network, or schedule. Treat `region` only as a coarse
   geographic label.
5. Fuel price changes do not mechanically change physical consumption. Adjust
   only when evidence plausibly changes flying activity, routing, payload,
   operational disruption, or efficiency.
6. Echo `external_model.forecast` exactly as `baseline_forecast`. Do not
   convert between minmax and indexed scales.

## Output

Return the structured response through `set_model_response`: the echoed
baseline, the adjusted forecast, `adjustment_pct` consistent with the two
forecasts, concise rationale, an evidence summary, and source URLs (empty when
no evidence was used).
"""


class FuelAdjustmentHorizon(BaseModel):
    """One horizon's external anchor and agent adjustment."""

    model_config = {"extra": "ignore"}

    horizon: int = Field(ge=1)
    baseline_forecast: float
    adjusted_forecast: float
    adjustment_pct: float
    rationale: str = ""
    evidence_summary: str = ""
    source_urls: list[str] = Field(default_factory=list)

    @field_validator("baseline_forecast", "adjusted_forecast", "adjustment_pct")
    @classmethod
    def finite_values(cls, value: float) -> float:
        """Reject NaN and infinite forecast values."""
        if not isfinite(value):
            raise ValueError("Forecast and adjustment values must be finite.")
        return value

    @model_validator(mode="after")
    def consistent_adjustment(self) -> "FuelAdjustmentHorizon":
        """Require a positive forecast and an internally consistent change.

        The cap is deliberately not enforced here: an over-cap response is
        clipped by :func:`clip_adjustments` so one aggressive answer yields a
        capped row instead of aborting the backtest.
        """
        if self.baseline_forecast <= 0 or self.adjusted_forecast <= 0:
            raise ValueError("Fuel-consumption forecasts must be positive.")
        expected = (self.adjusted_forecast / self.baseline_forecast - 1.0) * 100.0
        if not isclose(self.adjustment_pct, expected, abs_tol=0.05):
            raise ValueError(
                f"adjustment_pct ({self.adjustment_pct}) is inconsistent with "
                f"baseline/adjusted forecasts ({expected:.4f})."
            )
        return self


class FuelAdjustmentAgentOutput(AgentForecastOutput):
    """Structured point-adjustment output converted to standard predictions."""

    modality: ClassVar[Literal["continuous", "discrete", "categorical"]] = "continuous"
    model_config = {"extra": "ignore"}

    forecasts: list[FuelAdjustmentHorizon] = Field(min_length=1)
    overall_rationale: str = ""

    @model_validator(mode="after")
    def unique_horizons(self) -> "FuelAdjustmentAgentOutput":
        """Reject duplicate horizon entries."""
        horizons = [forecast.horizon for forecast in self.forecasts]
        if len(horizons) != len(set(horizons)):
            raise ValueError("Agent output contains duplicate horizons.")
        return self

    @classmethod
    def prompt_schema_json(cls) -> str:
        """Return a compact example of the exact expected response shape."""
        return json.dumps(
            {
                "forecasts": [
                    {
                        "horizon": "<integer>",
                        "baseline_forecast": "<number copied exactly from input>",
                        "adjusted_forecast": "<positive number>",
                        "adjustment_pct": "<number within +/- max_absolute_adjustment_pct>",
                        "rationale": "<why the adjustment is justified>",
                        "evidence_summary": "<cutoff-safe evidence or insufficiency statement>",
                        "source_urls": ["<https URL>"],
                    }
                ],
                "overall_rationale": "<brief synthesis>",
            },
            indent=2,
        )

    def to_predictions(
        self,
        *,
        task: ForecastingTask,
        context: ForecastContext,
        predictor_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Convert point adjustments into deterministic payloads."""
        expected_horizons = set(task.horizons)
        actual_horizons = {forecast.horizon for forecast in self.forecasts}
        if actual_horizons != expected_horizons:
            raise ValueError(
                f"Agent horizons must exactly match task horizons. "
                f"Expected {sorted(expected_horizons)}, got {sorted(actual_horizons)}."
            )

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        base_metadata = dict(metadata or {})
        predictions: list[Prediction] = []
        for forecast in sorted(self.forecasts, key=lambda item: item.horizon):
            prediction_metadata = {
                **base_metadata,
                "baseline_forecast": forecast.baseline_forecast,
                "adjustment_pct": forecast.adjustment_pct,
                "rationale": forecast.rationale,
                "evidence_summary": forecast.evidence_summary,
                "source_urls": forecast.source_urls,
                "overall_rationale": self.overall_rationale,
            }
            predictions.append(
                Prediction(
                    predictor_id=predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=(pd.Timestamp(context.as_of) + offset * forecast.horizon).to_pydatetime(),
                    payload=deterministic_payload(forecast.adjusted_forecast),
                    metadata=prediction_metadata,
                )
            )
        return predictions


def clip_adjustments(predictions: Sequence[Prediction], *, cap_pct: float) -> None:
    """Clip each point forecast to ``baseline * (1 +/- cap_pct/100)`` in place.

    ``adjustment_pct`` is always recomputed from the (possibly clipped) point so
    the recorded percentage and payload agree. The pre-clip percentage is kept
    as ``raw_adjustment_pct`` and ``cap_applied`` records whether the cap bound.
    """
    if cap_pct < 0:
        raise ValueError("cap_pct must be non-negative.")
    for prediction in predictions:
        baseline = float(prediction.metadata["baseline_forecast"])
        point = float(prediction.payload.point_forecast)  # type: ignore[union-attr]
        raw_pct = (point / baseline - 1.0) * 100.0
        capped = abs(raw_pct) > cap_pct
        if capped:
            pct = copysign(cap_pct, raw_pct)
            point = baseline * (1.0 + pct / 100.0)
            prediction.payload = deterministic_payload(point)
        else:
            pct = raw_pct
        prediction.metadata["raw_adjustment_pct"] = raw_pct
        prediction.metadata["adjustment_pct"] = pct
        prediction.metadata["cap_applied"] = capped
        prediction.metadata["cap_pct"] = float(cap_pct)


class FuelForecastPromptBuilder:
    """Build a ground-truth-free prompt around one external model forecast."""

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        forecast_scale: str,
        cap_pct: float = MAX_ABSOLUTE_ADJUSTMENT_PCT,
    ) -> None:
        self._scale = normalize_forecast_scale(forecast_scale)
        self._forecast_column = forecast_column(self._scale)
        self._cap_pct = float(cap_pct)
        self._data = forecast_input_frame(data)

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        """Serialize only information available to the adjustment pipeline."""
        if len(task.horizons) != 1:
            raise ValueError("AC One agent cases must contain exactly one horizon.")
        if scale_for_task(task) != self._scale:
            raise ValueError(f"Prompt scale {self._scale!r} does not match task series {task.target_series_id!r}.")
        horizon = task.horizons[0]
        station = station_for_task(task, self._data)
        row = lookup_external_forecast(
            self._data,
            station=station,
            origin=context.as_of,
            horizon=horizon,
        )
        payload = {
            "task": task.task_id,
            "as_of": str(pd.Timestamp(context.as_of).date()),
            "target_month": str(pd.Timestamp(row["horizon_date"]).date()),
            "station": station,
            "region": str(row["region"]),
            "month_horizon": horizon,
            "forecast_scale": self._scale,
            "external_model": {
                "forecast": float(row[self._forecast_column]),
                "model_version": str(row["model_version"]),
                "schedule_file_id": str(row["schd_file_id"]),
            },
            "constraints": {
                "max_absolute_adjustment_pct": self._cap_pct,
                "actual_is_unavailable": True,
                "do_not_infer_station_location": True,
                "do_not_convert_scale": True,
            },
            "output_schema": FuelAdjustmentAgentOutput.prompt_schema_json(),
        }
        return json.dumps(payload, indent=2)


def _langfuse_enabled(explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    try:
        import langfuse  # noqa: F401, PLC0415
    except ModuleNotFoundError:
        return False
    return True


def enabled_tool_labels(config: AgentConfig) -> list[str]:
    """Derive the toolbelt labels from the config so metadata cannot drift from it."""
    labels: list[str] = []
    if config.context_retrieval.enabled:
        labels.append("news_search")
    labels.extend(str(getattr(tool, "name", type(tool).__name__)) for tool in config.function_tools)
    return labels


class FuelAdjustmentPredictor(AgentPredictor):
    """Agent predictor that clips the adjustment and verifies the echoed anchor."""

    def __init__(
        self,
        data: pd.DataFrame,
        agent_config: AgentConfig,
        *,
        forecast_scale: str,
        cap_pct: float = MAX_ABSOLUTE_ADJUSTMENT_PCT,
        enable_langfuse_tracing: bool | None = None,
    ) -> None:
        self._scale: ForecastScale = normalize_forecast_scale(forecast_scale)
        self._forecast_column = forecast_column(self._scale)
        self._cap_pct = float(cap_pct)
        self._tools_enabled = enabled_tool_labels(agent_config)
        self._forecast_data = forecast_input_frame(data)
        tracing = _langfuse_enabled(enable_langfuse_tracing)
        built_agent = build_adk_agent(agent_config, output_schema=FuelAdjustmentAgentOutput)
        runner = AdkTextRunner(
            agent=built_agent,
            config=AdkTextRunnerConfig(
                app_name="agentic_forecasting_predictor",
                default_user_id="forecasting_agent",
                fresh_session_per_message=True,
                enable_langfuse_tracing=tracing,
                langfuse_tags=[
                    "agent_predictor",
                    "track1",
                    LANGFUSE_PROJECT_NAME,
                    "ac_one",
                    f"scale_{self._scale}",
                ],
                langfuse_trace_name=f"agent_predictor_{built_agent.name}",
                # Langfuse drops non-ASCII propagate values; keep these plain strings
                # and never put a real station name here.
                langfuse_propagate_metadata={
                    "predictor_id": f"agent_predictor_{built_agent.name}",
                    "agent_name": built_agent.name,
                    "model": str(built_agent.model),
                    "output_modality": "continuous",
                    "forecast_scale": self._scale,
                    "cap_pct": f"{self._cap_pct:g}",
                    "tools": ",".join(self._tools_enabled) or "none",
                    "langfuse_project": LANGFUSE_PROJECT_NAME,
                },
            ),
        )
        super().__init__(
            agent_config=agent_config,
            prompt_builder=FuelForecastPromptBuilder(data, forecast_scale=self._scale, cap_pct=self._cap_pct),
            output_schema=FuelAdjustmentAgentOutput,
            enable_langfuse_tracing=tracing,
            runner=runner,
        )

    @property
    def cap_pct(self) -> float:
        """Absolute percentage bound applied to every adjustment in Python."""
        return self._cap_pct

    @property
    def tools_enabled(self) -> list[str]:
        """Toolbelt labels derived from the agent config."""
        return list(self._tools_enabled)

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Run the agent, clip to the cap, reject a changed anchor, and stamp the trace."""
        if scale_for_task(task) != self._scale:
            raise ValueError(f"Predictor scale {self._scale!r} does not match task series {task.target_series_id!r}.")
        predictions = super().predict(task, context)
        clip_adjustments(predictions, cap_pct=self._cap_pct)
        self._verify_and_enrich(predictions, task=task, context=context)
        # The shared stamp covers categorical forecasts only, so stamp the adjusted
        # point forecast here — after enrichment, so the station, scale, and horizon
        # the trace evaluator joins on reach Langfuse rather than only the local cache.
        stamp_continuous_forecasts(predictions, trace_id=self._runner.last_trace_id)
        return predictions

    def _verify_and_enrich(
        self,
        predictions: list[Prediction],
        *,
        task: ForecastingTask,
        context: ForecastContext,
    ) -> None:
        """Verify the echoed anchor and attach the identity the trace evaluator needs."""
        station = station_for_task(task, self._forecast_data)
        horizon = task.horizons[0]
        row = lookup_external_forecast(
            self._forecast_data,
            station=station,
            origin=context.as_of,
            horizon=horizon,
        )
        expected = float(row[self._forecast_column])
        for prediction in predictions:
            echoed = float(prediction.metadata["baseline_forecast"])
            if not isclose(echoed, expected, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(f"Agent changed the external baseline: expected {expected}, echoed {echoed}.")
            prediction.metadata["forecast_scale"] = self._scale
            prediction.metadata["station"] = station
            prediction.metadata["region"] = str(row["region"])
            prediction.metadata["month_horizon"] = horizon
            prediction.metadata["external_forecast"] = expected
            prediction.metadata["tools_enabled"] = list(self._tools_enabled)


def build_fuel_adjustment_config(
    model: str = LITE_MODEL,
    *,
    forecast_scale: str,
    tools: Sequence[ToolSpec] | None = None,
) -> AgentConfig:
    """Fold a toolbelt onto the fuel-consumption adjustment agent for one scale.

    ``tools=None`` (the default) means ``[news_search()]`` so existing call
    sites keep the news-grounded agent. Pass ``tools=()`` for the no-evidence
    control, which must return zero adjustments.
    """
    scale = normalize_forecast_scale(forecast_scale)
    toolbelt = [news_search()] if tools is None else list(tools)

    skills_dirs: list[Path] = []
    instruction = _ANALYST_INSTRUCTION
    context_retrieval = ContextRetrievalConfig()
    function_tools: list[Any] = []
    max_output_tokens: int | None = None
    for spec in toolbelt:
        if spec.skill_dir is not None:
            skills_dirs.append(spec.skill_dir)
        if spec.instruction_supplement:
            instruction += spec.instruction_supplement
        if spec.context_retrieval is not None:
            context_retrieval = spec.context_retrieval
        if spec.function_tool is not None:
            function_tools.append(spec.function_tool)
        if spec.max_output_tokens is not None:
            max_output_tokens = max(max_output_tokens or 0, spec.max_output_tokens)

    return AgentConfig(
        name=f"ac_one_fuel_news_adjuster_{scale}",
        model=model,
        instruction=instruction,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        context_retrieval=context_retrieval,
        function_tools=function_tools,
        skills_dirs=skills_dirs,
    )


def build_fuel_adjustment_predictor(
    data: pd.DataFrame,
    *,
    forecast_scale: str,
    config: AgentConfig | None = None,
    tools: Sequence[ToolSpec] | None = None,
    cap_pct: float = MAX_ABSOLUTE_ADJUSTMENT_PCT,
    enable_langfuse_tracing: bool | None = None,
) -> FuelAdjustmentPredictor:
    """Wrap the AC One configuration in the shared agent predictor.

    ``tools`` is only consulted when ``config`` is not supplied.
    """
    resolved_config = config or build_fuel_adjustment_config(forecast_scale=forecast_scale, tools=tools)
    return FuelAdjustmentPredictor(
        data,
        resolved_config,
        forecast_scale=forecast_scale,
        cap_pct=cap_pct,
        enable_langfuse_tracing=enable_langfuse_tracing,
    )


def __getattr__(name: str) -> Any:
    """Expose a lazy interactive agent for ADK CLI/web."""
    if name == "root_agent":
        return build_adk_agent(build_fuel_adjustment_config(forecast_scale="indexed"))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LANGFUSE_PROJECT_NAME",
    "MAX_ABSOLUTE_ADJUSTMENT_PCT",
    "FuelAdjustmentAgentOutput",
    "FuelAdjustmentHorizon",
    "FuelAdjustmentPredictor",
    "FuelForecastPromptBuilder",
    "ToolSpec",
    "build_fuel_adjustment_config",
    "build_fuel_adjustment_predictor",
    "clip_adjustments",
    "enabled_tool_labels",
    "news_search",
]

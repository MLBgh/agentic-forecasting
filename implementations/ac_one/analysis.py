"""Point-forecast evaluation helpers for the AC One comparison."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from ac_one.data import actual_column, normalize_forecast_scale
from ac_one.predictors import scale_for_task, station_for_task
from aieng.forecasting.evaluation import BacktestResult, ContinuousForecast


def row_target_months(data: pd.DataFrame) -> pd.Series:
    """Return the target month for each forecast row.

    ``horizon_date`` is the forecast origin. The realized month is origin plus
    ``month_horizon``, unless the frame already carries ``target_month``.
    """
    if "target_month" in data.columns:
        return pd.to_datetime(data["target_month"])
    origins = pd.to_datetime(data["horizon_date"])
    return pd.Series(
        [
            origin + pd.DateOffset(months=int(horizon))
            for origin, horizon in zip(origins, data["month_horizon"], strict=True)
        ],
        index=data.index,
    )


def predictions_to_frame(
    results_by_predictor: dict[str, dict[str, BacktestResult]],
    data: pd.DataFrame,
    *,
    forecast_scale: str | None = None,
) -> pd.DataFrame:
    """Flatten results and attach matching evaluation-only actuals, MAE, and APE.

    Actuals are joined on station and target month (origin ``horizon_date`` plus
    ``month_horizon``), not on ``horizon_date`` itself.
    """
    rows: list[dict[str, Any]] = []

    for predictor_name, case_results in results_by_predictor.items():
        for case_name, result in case_results.items():
            station = station_for_task(result.spec.task, data)
            scale = forecast_scale or scale_for_task(result.spec.task)
            resolved_scale = normalize_forecast_scale(scale)
            actual_col = actual_column(resolved_scale)
            lookup_frame = data.assign(_target_month=row_target_months(data))
            actual_lookup = (
                lookup_frame[["station", "region", "_target_month", actual_col]]
                .drop_duplicates(["station", "_target_month"])
                .set_index(["station", "_target_month"])
            )
            horizon = result.spec.task.horizons[0]
            for prediction in result.predictions:
                if not isinstance(prediction.payload, ContinuousForecast):
                    raise TypeError("AC One analysis requires continuous point forecasts.")
                target = pd.Timestamp(prediction.forecast_date)
                try:
                    actual_row = actual_lookup.loc[(station, target)]
                except KeyError as exc:
                    raise ValueError(f"No actual found for station={station!r}, target={target.date()}.") from exc
                actual = float(actual_row[actual_col])
                if actual == 0:
                    raise ValueError(
                        f"MAPE is undefined for zero actual at station={station!r}, target={target.date()}."
                    )
                forecast = float(prediction.payload.point_forecast)
                absolute_error = abs(forecast - actual)
                rows.append(
                    {
                        "predictor": predictor_name,
                        "case": case_name,
                        "forecast_scale": resolved_scale,
                        "station": station,
                        "region": str(actual_row["region"]),
                        "origin": pd.Timestamp(prediction.as_of),
                        "target_month": target,
                        "horizon": horizon,
                        "forecast": forecast,
                        "actual": actual,
                        "absolute_error": absolute_error,
                        "ape_pct": absolute_error / abs(actual) * 100.0,
                        "adjustment_pct": prediction.metadata.get("adjustment_pct", 0.0),
                        "baseline_forecast": prediction.metadata.get("baseline_forecast", forecast),
                        "rationale": prediction.metadata.get("rationale", ""),
                        "evidence_summary": prediction.metadata.get("evidence_summary", ""),
                        "source_urls": prediction.metadata.get("source_urls", []),
                        "langfuse_trace_id": prediction.metadata.get("langfuse_trace_id", ""),
                        "langfuse_trace_url": prediction.metadata.get("langfuse_trace_url", ""),
                    }
                )

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["forecast_scale", "predictor", "station", "target_month", "horizon"])
        .reset_index(drop=True)
    )


def metric_display_formats(scale: str | None = None) -> dict[str, str]:
    """Return pandas styler formats that keep small minmax MAE visible."""
    mae = "{:.4f}" if scale == "minmax" else "{:.3f}"
    return {"mae": mae, "mape_pct": "{:.2f}", "n_forecasts": "{:,.0f}"}


def improvement_display_formats(scale: str | None = None) -> dict[str, str]:
    """Return pandas styler formats for paired improvement tables."""
    mae = "{:.4f}" if scale == "minmax" else "{:.3f}"
    return {
        "mean_mae_improvement": mae,
        "mean_mape_improvement_pct_points": "{:.2f}",
        "win_rate_pct": "{:.1f}",
        "n_pairs": "{:,.0f}",
    }


def metric_summary(
    scored: pd.DataFrame,
    *,
    by: Sequence[str] = (),
) -> pd.DataFrame:
    """Aggregate MAE and MAPE by predictor and optional dimensions."""
    required = {"predictor", "absolute_error", "ape_pct"}
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"Scored frame is missing required columns: {missing}")
    group_columns = ["predictor", *by]
    if "forecast_scale" in scored.columns and "forecast_scale" not in group_columns:
        group_columns = ["forecast_scale", *group_columns]
    return (
        scored.groupby(group_columns, dropna=False)
        .agg(
            mae=("absolute_error", "mean"),
            mape_pct=("ape_pct", "mean"),
            n_forecasts=("absolute_error", "size"),
        )
        .reset_index()
        .sort_values(
            ["forecast_scale", "mape_pct", "mae"] if "forecast_scale" in group_columns else ["mape_pct", "mae"]
        )
        .reset_index(drop=True)
    )


def paired_improvement(
    scored: pd.DataFrame,
    *,
    baseline: str = "External XGBoost",
    adjusted: str = "News-adjusted agent",
    by: Sequence[str] = (),
) -> pd.DataFrame:
    """Compare adjusted and baseline errors on identical forecast cases."""
    key_columns = ["forecast_scale", "station", "origin", "target_month", "horizon"]
    if "forecast_scale" not in scored.columns:
        key_columns = ["station", "origin", "target_month", "horizon"]
    metric_columns = ["absolute_error", "ape_pct"]
    subset = scored[scored["predictor"].isin([baseline, adjusted])]
    pivot = subset.pivot(index=key_columns, columns="predictor", values=metric_columns)
    required_columns = {
        ("absolute_error", baseline),
        ("absolute_error", adjusted),
        ("ape_pct", baseline),
        ("ape_pct", adjusted),
    }
    if not required_columns.issubset(set(pivot.columns)):
        missing = sorted(required_columns - set(pivot.columns))
        raise ValueError(f"Paired comparison is missing predictor columns: {missing}")

    paired = pivot.reset_index()
    paired.columns = [
        str(column[0]) if not column[1] else f"{column[0]}__{column[1]}" for column in paired.columns.to_flat_index()
    ]
    baseline_mae = f"absolute_error__{baseline}"
    adjusted_mae = f"absolute_error__{adjusted}"
    baseline_mape = f"ape_pct__{baseline}"
    adjusted_mape = f"ape_pct__{adjusted}"
    paired["mae_improvement"] = paired[baseline_mae] - paired[adjusted_mae]
    paired["mape_improvement_pct_points"] = paired[baseline_mape] - paired[adjusted_mape]
    paired["agent_win"] = paired[adjusted_mae] < paired[baseline_mae]

    group_columns = list(by)
    if "forecast_scale" in paired.columns and "forecast_scale" not in group_columns:
        group_columns = ["forecast_scale", *group_columns]
    if group_columns:
        grouped = paired.groupby(group_columns, dropna=False)
        summary = grouped.agg(
            mean_mae_improvement=("mae_improvement", "mean"),
            mean_mape_improvement_pct_points=("mape_improvement_pct_points", "mean"),
            win_rate=("agent_win", "mean"),
            n_pairs=("agent_win", "size"),
        ).reset_index()
    else:
        summary = pd.DataFrame(
            [
                {
                    "mean_mae_improvement": float(paired["mae_improvement"].mean()),
                    "mean_mape_improvement_pct_points": float(paired["mape_improvement_pct_points"].mean()),
                    "win_rate": float(paired["agent_win"].mean()),
                    "n_pairs": int(len(paired)),
                }
            ]
        )
    summary["win_rate_pct"] = summary.pop("win_rate") * 100.0
    return summary


def mean_absolute_error(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Compute MAE with explicit shape validation."""
    actual_array = np.asarray(actual, dtype=float)
    forecast_array = np.asarray(forecast, dtype=float)
    if actual_array.shape != forecast_array.shape:
        raise ValueError("actual and forecast must have the same shape.")
    if actual_array.size == 0:
        return float("nan")
    return float(np.mean(np.abs(forecast_array - actual_array)))


def mean_absolute_percentage_error(
    actual: Sequence[float],
    forecast: Sequence[float],
) -> float:
    """Compute MAPE in percent, rejecting zero actuals."""
    actual_array = np.asarray(actual, dtype=float)
    forecast_array = np.asarray(forecast, dtype=float)
    if actual_array.shape != forecast_array.shape:
        raise ValueError("actual and forecast must have the same shape.")
    if actual_array.size == 0:
        return float("nan")
    if np.any(actual_array == 0):
        raise ValueError("MAPE is undefined when any actual value is zero.")
    return float(np.mean(np.abs(forecast_array - actual_array) / np.abs(actual_array)) * 100.0)


__all__ = [
    "improvement_display_formats",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "metric_display_formats",
    "metric_summary",
    "paired_improvement",
    "predictions_to_frame",
    "row_target_months",
]

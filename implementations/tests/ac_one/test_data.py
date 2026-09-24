"""Tests for AC One scale-aware loading and series registration."""

from __future__ import annotations

import pandas as pd
from ac_one.data import (
    FORECAST_SCALES,
    REQUIRED_COLUMNS,
    actual_column,
    actuals_frame,
    build_ac_one_service,
    forecast_column,
    load_forecast_data,
    station_series_id,
    target_month_for,
)


def test_checked_in_csv_matches_required_schema() -> None:
    """The committed export has paired scaled columns and no unsuffixed actual/forecast."""
    data = load_forecast_data()
    assert REQUIRED_COLUMNS.issubset(set(data.columns))
    assert "forecast" not in data.columns
    assert "actual" not in data.columns
    assert len(data) == 30


def test_target_month_is_the_run_month_plus_the_lead() -> None:
    """A lead of 2 issued in January targets March; horizon_date is the origin."""
    origin = pd.Timestamp("2026-01-01")
    assert target_month_for(origin, 1) == pd.Timestamp("2026-02-01")
    assert target_month_for(origin, 2) == pd.Timestamp("2026-03-01")
    data = load_forecast_data()
    assert (data["forecast_origin"] == data["horizon_date"]).all()


def test_actuals_are_not_visible_until_the_month_has_ended() -> None:
    """A run at the start of month M knows actuals only through M - 1."""
    data = load_forecast_data()
    service = build_ac_one_service(data)
    for row in data.itertuples():
        origin = pd.Timestamp(row.forecast_origin)
        context = service.context(origin.to_pydatetime())
        for scale in FORECAST_SCALES:
            visible = context.get_series(station_series_id(str(row.station), scale))
            assert (visible["timestamp"] < origin).all()


def test_scale_column_pairs() -> None:
    """Each scale maps to its matching forecast and actual columns."""
    assert forecast_column("minmax") == "forecast_minmax"
    assert actual_column("minmax") == "actual_minmax"
    assert forecast_column("indexed") == "forecast_indexed"
    assert actual_column("indexed") == "actual_indexed"


def test_series_ids_include_scale_and_register_separately() -> None:
    """DataService series IDs isolate minmax and indexed actuals."""
    data = load_forecast_data()
    service = build_ac_one_service(data)
    station = str(data["station"].iloc[0])
    minmax_id = station_series_id(station, "minmax")
    indexed_id = station_series_id(station, "indexed")
    assert minmax_id != indexed_id
    assert minmax_id.endswith("_minmax")
    assert indexed_id.endswith("_indexed")
    as_of = pd.Timestamp("2026-12-01").to_pydatetime()
    minmax_values = service.get_series(minmax_id, as_of)["value"].to_numpy()
    indexed_values = service.get_series(indexed_id, as_of)["value"].to_numpy()
    assert not (minmax_values == indexed_values).all()
    for scale in FORECAST_SCALES:
        expected = actuals_frame(data, station, scale)["value"].to_numpy()
        registered = service.get_series(station_series_id(station, scale), as_of)["value"].to_numpy()
        assert list(expected) == list(registered)

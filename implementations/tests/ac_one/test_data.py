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
)


def test_checked_in_csv_matches_required_schema() -> None:
    """The committed export has paired scaled columns and no unsuffixed actual/forecast."""
    data = load_forecast_data()
    assert REQUIRED_COLUMNS.issubset(set(data.columns))
    assert "forecast" not in data.columns
    assert "actual" not in data.columns
    assert len(data) == 36


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

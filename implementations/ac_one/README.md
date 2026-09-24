# AC One Fuel Consumption Forecasting

The **external-forecast adjustment** reference implementation. It forecasts
monthly airline fuel-consumption volume per anonymized station at 1-, 2-, and
3-month horizons.

The pipeline begins after an existing XGBoost model has run. That model remains
an external module: its point forecasts arrive in the supplied CSV on two
scales, and a cutoff-aware news agent decides whether exceptional public
information justifies a conservative adjustment. Each adjusted forecast is
compared with the matching evaluation-only actual using MAE and MAPE.

## Pipeline

```text
forecast_minmax  → baseline / news agent → MAE / MAPE vs actual_minmax
forecast_indexed → baseline / news agent → MAE / MAPE vs actual_indexed
```

The agent is an overlay, not a replacement model. It does not recreate the
XGBoost feature pipeline and does not receive actual consumption. Run both
scales through the same agent so you can see which external-forecast input
produces better results. MAE is scale-specific; use MAPE to compare scales.

## Data contract

[`VECTOR___AGENTIC_FORECASTING_DATA.csv`](VECTOR___AGENTIC_FORECASTING_DATA.csv)
contains 30 rows: two anonymized stations, a ragged origin panel (six / five /
four origins at 1- / 2- / 3-month leads), and target months derived as origin
plus lead.

- `horizon_date` is the forecast origin (the month the forecast was issued).
- `month_horizon` is the lead in months.
- the target month is `horizon_date` plus `month_horizon` months.
- `forecast_minmax` / `actual_minmax` are one paired evaluation scale.
- `forecast_indexed` / `actual_indexed` are the other paired evaluation scale.
- There is no unsuffixed `forecast` or `actual` column.
- `station` and `region` remain coarse anonymized labels. The agent is forbidden
  from inferring a specific airport, city, or route network.

The loader in [`data.py`](data.py) validates columns, nulls, key uniqueness,
month alignment, horizon positivity, and agreement of each scale's actuals
across origins that share a **target month**. It registers one actual series per station **and scale** with
the shared `DataService`. Cache and spec IDs include the scale so the two
runs cannot overwrite each other.

## Forecasting and leakage controls

[`predictors.py`](predictors.py) adapts the selected-scale external forecast to
the shared `Predictor`/`Prediction` contract (Case A, the unadjusted baseline).
[`analyst_agent/agent.py`](analyst_agent/agent.py) is Case B: the common ADK
`AgentPredictor` with a toolbelt, running at temperature 0 on Vector-proxy models.

For every historical issue date:

1. the prompt includes only station, region, issue/target dates, horizon,
   forecast scale, that scale's external forecast, and non-outcome model metadata
   (`forecast_input_frame()` drops `actual_*` before anything reaches the builder);
2. `search_web` is called with the issue date as its cutoff, and the harness
   seeds the same date into the session so the tool enforces it regardless of
   what the model passes;
3. the independent verifier rejects post-cutoff claims;
4. the agent must leave the forecast unchanged when evidence is weak, and with
   an empty toolbelt it has no evidence at all;
5. the adjustment is clipped in Python to ±`cap_pct` (default 20%) after the
   response is parsed — the prompt states the cap, but the code enforces it;
6. the predictor rejects any response whose echoed baseline differs from the CSV;
7. the agent must not convert between minmax and indexed values.

### Toolbelt seam

An agent is the fixed analyst instruction plus a list of `ToolSpec`s from
[`analyst_agent/tools.py`](analyst_agent/tools.py), folded by
`build_fuel_adjustment_config(forecast_scale=..., tools=...)`:

```python
from ac_one.analyst_agent import build_fuel_adjustment_config, news_search

config = build_fuel_adjustment_config(forecast_scale="indexed")                  # default: [news_search()]
config = build_fuel_adjustment_config(forecast_scale="indexed", tools=())        # no-evidence control
config = build_fuel_adjustment_config(forecast_scale="indexed", tools=[news_search(search_model=...)])
```

`news_search()` wraps the shared `search_web` sub-agent (proxy Google Search
plus the post-cutoff verifier) and loads the
[`news-adjustment`](analyst_agent/skills/news-adjustment/SKILL.md) skill. Search
behaviour is edited in exactly two places — the search instruction in
`tools.py` and that skill — never in the evaluation code. Search queries must
not contain real station names or CSV actuals.

`Prediction.metadata` records `cap_pct`, `tools_enabled`, `raw_adjustment_pct`
and `cap_applied`, and the Langfuse trace metadata carries `cap_pct` and `tools`
as ASCII strings, so a result is always attributable to a specific toolbelt.

Relevant evidence includes capacity or demand changes, cancellations, airspace
or routing disruption, severe weather, and operationally material policy
changes. Fuel-price news alone is not treated as a direct change in physical
consumption.

## Evaluation

[`00_smoke_adjustment.ipynb`](00_smoke_adjustment.ipynb) is the cheap check:
three indexed rows, `tools=()` then `tools=[news_search()]`, with an offline
proof that the prompt has no `actual_*` key, the baseline echoes the CSV, and
the cap clips. The live cells run only when the proxy key is present.

Run [`01_agentic_forecast_adjustment.ipynb`](01_agentic_forecast_adjustment.ipynb)
for the comparison.
The default `RUN_AGENT = False` makes “Run All” free: it evaluates both external
baselines and loads any cached agent artifacts. Set it to `True` deliberately to
make 30 agent calls per scale plus search/verifier calls.

The primary metrics are:

- **MAE** — average absolute error, in that scale's anonymized units. Do not
  rank minmax against indexed using MAE.
- **MAPE** — average absolute percentage error, used to compare stations,
  horizons, and the two input scales. Zero actuals are rejected because MAPE
  would be undefined.

[`analysis.py`](analysis.py) and [`trace_eval.py`](trace_eval.py) join each
prediction to its actual on station and **target month** (`horizon_date` +
`month_horizon`). `horizon_date` is the origin, not the month being scored.

If the agent copies the baseline, paired MAE/MAPE *improvement* is exactly 0.
That is a real outcome, not a missing-actual bug.

The shared harness requires continuous payloads, so both methods store their
point forecasts as deterministic quantile payloads for artifact compatibility.
CRPS is therefore not used to rank this point-only comparison.

[`analysis.py`](analysis.py) reports overall, station, region, and horizon
breakdowns plus paired error improvement and win rate, grouped by forecast
scale. With only 30 rows per scale, these results are descriptive; reserve
newer months as a protected window before operational tuning.

## Langfuse (project `air-canada-1`)

Traces are sent to the Langfuse project that issued your API keys. For this
use case that project is **`air-canada-1`**. Set these in the environment or
repo `.env` (never commit secrets):

```text
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

`LANGFUSE_HOST` must match the project's region. The notebook calls
`connect_langfuse()` and `auth_check()` before agent runs. Agent traces are
tagged `air-canada-1`, `ac_one`, and `scale_minmax` / `scale_indexed`.

`LANGFUSE_PROJECT_NAME` is a label used for those tags, the score metadata, and
the notebook's status messages — it does not route anything. The destination
project is whichever one issued the keys, so the project shown in the Langfuse
UI can differ from the constant without anything being misconfigured.

[`trace_stamp.py`](trace_stamp.py) writes each adjusted point forecast onto the
`forecast` observation of its trace. The predictor stamps only after it has
verified the echoed anchor, so the station, forecast scale, and horizon that
scoring joins on are already attached rather than living only in the local cache.

With `RUN_TRACE_EVAL = True`, [`trace_eval.py`](trace_eval.py) reads each
stamped forecast from Langfuse, joins the matching `actual_*` column, and
pushes deterministic scores (`absolute_error`, `ape_pct`, improvements,
`adjustment_pct`) back onto the same traces. Set `RUN_RATIONALE_JUDGE = True`
only when you want extra LLM-as-judge calls for rationale quality.

Traces stamped by earlier runs can carry the station, scale, and horizon empty;
`task_identity_lookup()` recovers them from the stamped `task_id`, so
already-recorded traces stay scoreable without re-running the agent.
`max_wait_s` bounds the per-trace readiness poll — the default suits
already-ingested traces, and every unresolvable trace ID costs that full
budget, so raise it only when scoring a run that just finished.

## Core library boundary

Everything this use case adds lives under `implementations/ac_one/`, with its
tests under `implementations/tests/ac_one/`. The shared `aieng.forecasting`
library is used as-is — the `Predictor`/`Prediction` contracts, `AgentPredictor`
and the ADK runner, the Langfuse trace reader and scorer — and anything the
overlay needs beyond it is implemented here rather than by extending the
library: [`predictors.py`](predictors.py) adapts the external forecasts to the
shared contract, and [`trace_stamp.py`](trace_stamp.py) supplies the continuous
forecast payload that the library's categorical-only
`stamp_forecast_on_trace()` does not write.

## Layout

```text
implementations/ac_one/
├── VECTOR___AGENTIC_FORECASTING_DATA.csv
├── data.py
├── specs.py
├── predictors.py
├── analysis.py
├── trace_stamp.py
├── trace_eval.py
├── analyst_agent/
│   ├── agent.py                 # Case B predictor, config fold, Python cap
│   ├── tools.py                 # ToolSpec + news_search() (search instruction lives here)
│   └── skills/news-adjustment/SKILL.md
├── specs/ac_one_backtest.yaml
├── 00_smoke_adjustment.ipynb    # 3-row seam smoke
└── 01_agentic_forecast_adjustment.ipynb
```

`tests/ac_one/` covers the loader, Case A lookup, MAE/MAPE scoring, the trace
scorer, the toolbelt fold, the Python cap, and prompt leakage.

## Extensions

- Add schedule/capacity features or forecast metadata that can reveal whether a
  news event is already represented by the external model.
- Calibrate `cap_pct` by station and horizon on a development window.
- Add a second `ToolSpec` (for example structured price or capacity signals)
  next to `news_search()`; the fold and the metadata already accommodate it.
- Record prospective forecasts and evaluate them only after each target month
  resolves.
- Add a true probabilistic forecast from the external model; only then compare
  both methods with CRPS and interval calibration.

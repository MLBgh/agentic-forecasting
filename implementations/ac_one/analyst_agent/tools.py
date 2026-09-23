"""The AC One adjustment agent's toolbelt — one factory per tool.

Each factory returns a :class:`ToolSpec` that
:func:`~ac_one.analyst_agent.agent.build_fuel_adjustment_config` folds onto a
single ``AgentConfig``::

    from ac_one.analyst_agent import build_fuel_adjustment_config, tools

    config = build_fuel_adjustment_config(
        forecast_scale="indexed",
        tools=[tools.news_search()],  # cutoff-verified proxy web search (default)
    )

``news_search`` is the only tool today. It wraps the shared ``search_web``
sub-agent (Google Search through the Vector proxy plus an independent
post-cutoff verifier); the search instruction and the ``news-adjustment``
skill next to this module are the two places to edit search behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aieng.forecasting.methods.agentic.agent_factory import ContextRetrievalConfig
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL


_SKILLS_ROOT = Path(__file__).parent / "skills"
_NEWS_SKILL = _SKILLS_ROOT / "news-adjustment"


@dataclass(frozen=True)
class ToolSpec:
    """One item on the agent's toolbelt.

    Attributes
    ----------
    label : str
        Short name recorded in ``Prediction.metadata["tools_enabled"]`` and
        Langfuse metadata.
    context_retrieval : ContextRetrievalConfig or None
        Web-search sub-agent config, if this tool provides search.
    function_tool : Any or None
        A ready-to-register ADK function tool.
    skill_dir : Path or None
        A skill directory loaded alongside the tool.
    instruction_supplement : str
        Text appended to the analyst instruction when this tool is on.
    max_output_tokens : int or None
        Per-tool floor on the response budget; the fold takes the max.
    """

    label: str
    context_retrieval: ContextRetrievalConfig | None = None
    function_tool: Any | None = None
    skill_dir: Path | None = None
    instruction_supplement: str = ""
    max_output_tokens: int | None = None


# Station names are anonymized in every prompt, log, and trace. Nothing here may
# name an airport, city, or carrier; region is the only geographic hint allowed.
_SEARCH_INSTRUCTION = """\
You are an aviation market-intelligence researcher with web search.

Return a concise, source-grounded briefing relevant to airline physical fuel
consumption. Prioritize evidence about aviation capacity and demand,
cancellations or airspace/routing disruption, major regional weather, jet-fuel
or crude-market shocks that affect operations, and material policy changes.
Distinguish price effects from physical-consumption effects.

The station name is anonymized, so do not infer a specific airport or city.
When a cutoff is supplied, exclude every event or claim that became known after
that date. Judge recency from the substance of each result, not from a claimed
publish date. Do not fill gaps from background knowledge. Include source URLs
and say explicitly when evidence is insufficient.
"""

_NEWS_SEARCH_SUPPLEMENT = """

## Web search tool

You have `search_web`. Load the `news-adjustment` skill and follow it.

- Call `search_web` at least once before answering, with `cutoff_date` equal
  to the payload `as_of` date.
- Use only evidence dated on or before that cutoff. Evidence the verifier
  rejects does not exist for this task.
- Search for airline/airport operating conditions, aviation demand or
  capacity, jet-fuel and energy developments, geopolitical or airspace
  disruption, and major regional weather. Do not search for the station's
  identity.
- List the source URLs you relied on in `source_urls`.\
"""


def news_search(
    *,
    search_model: str = LITE_MODEL,
    verifier_model: str = ADVANCED_MODEL,
    instruction: str = _SEARCH_INSTRUCTION,
) -> ToolSpec:
    """Build the cutoff-verified web-search tool (proxy-only, no extra key).

    Parameters
    ----------
    search_model : str
        Model for the search sub-agent.
    verifier_model : str
        Model for the independent post-cutoff verifier. Kept distinct from
        ``search_model`` so the verifier does not share the searcher's blind spots.
    instruction : str
        System instruction for the search sub-agent.
    """
    return ToolSpec(
        label="news_search",
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            search_model=search_model,
            verifier_model=verifier_model,
            instruction=instruction,
        ),
        skill_dir=_NEWS_SKILL,
        instruction_supplement=_NEWS_SEARCH_SUPPLEMENT,
    )


__all__ = ["ToolSpec", "news_search"]

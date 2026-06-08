"""Base types for the genome feature annotation agent pipeline.

AgentContext carries the inputs to any agent. Agent is a structural protocol —
any class with a `name` attribute and a `run(ctx) -> dict` method satisfies it.
run_pipeline chains agents sequentially, threading prior outputs through context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from genome_feature_atlas.annotation_summary import FeatureSummary


@dataclass
class AgentContext:
    """Inputs shared by every agent in the pipeline."""

    feature_summary: FeatureSummary
    background_text: str                            # formatted by _fmt_background() in prototype_summary.py
    prior_outputs: dict[str, dict] = field(default_factory=dict)  # agent_name → output dict


@runtime_checkable
class Agent(Protocol):
    name: str

    def run(self, ctx: AgentContext) -> dict: ...


def run_pipeline(agents: list[Agent], ctx: AgentContext) -> dict[str, dict]:
    """Run agents in sequence; each receives the accumulated outputs of prior agents.

    Returns a dict mapping agent name → output dict.
    """
    outputs: dict[str, dict] = {}
    for agent in agents:
        result = agent.run(ctx)
        outputs[agent.name] = result
        ctx = AgentContext(
            feature_summary=ctx.feature_summary,
            background_text=ctx.background_text,
            prior_outputs={**ctx.prior_outputs, agent.name: result},
        )
    return outputs

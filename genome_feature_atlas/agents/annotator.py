"""AnnotatorAgent: SAE feature → structured JSON annotation via local LLM."""

from __future__ import annotations

import json
import re

from genome_feature_atlas.agents.base import AgentContext
from genome_feature_atlas.agents.llm import LLMClient
from genome_feature_atlas.agents.prompts import ANNOTATOR_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES


class AnnotatorAgent:
    """Annotates a single SAE feature using the local vLLM instance.

    The agent builds a user prompt from an optional few-shot header, the background
    text, and the feature summary, then parses the model's JSON response.

    Args:
        system_prompt: overrides ANNOTATOR_SYSTEM_PROMPT for A/B testing.
        client:        custom LLMClient; defaults to a new thinking-enabled client.
        few_shot:      prepend FEW_SHOT_EXAMPLES to the user message (default True).
    """

    name = "annotator"

    def __init__(
        self,
        system_prompt: str | None = None,
        client: LLMClient | None = None,
        few_shot: bool = True,
    ) -> None:
        self._system = system_prompt or ANNOTATOR_SYSTEM_PROMPT
        self._client = client or LLMClient(system=self._system, thinking=True)
        self._few_shot = few_shot

    def run(self, ctx: AgentContext) -> dict:
        """Return a parsed annotation dict for the feature in ctx."""
        prompt = self._build_prompt(ctx)
        raw = self._client.ask(prompt)
        return _parse_json(raw)

    def _build_prompt(self, ctx: AgentContext) -> str:
        parts: list[str] = []
        if self._few_shot and FEW_SHOT_EXAMPLES:
            parts.append(_fmt_few_shot())
        parts.append(ctx.background_text)
        parts.append(ctx.feature_summary.to_text())
        return "\n\n".join(parts)


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_few_shot() -> str:
    """Format FEW_SHOT_EXAMPLES as inline Q/A pairs for the user message."""
    lines = ["The following are example annotations to calibrate your output format and reasoning:"]
    for feature_text, annotation_json in FEW_SHOT_EXAMPLES:
        lines.append("")
        lines.append("FEATURE:")
        lines.append(feature_text.strip())
        lines.append("ANNOTATION:")
        lines.append(annotation_json.strip())
    lines.append("")
    lines.append("Now annotate the following feature:")
    return "\n".join(lines)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json(raw: str) -> dict:
    """Extract and parse the JSON object from raw model output.

    Strips markdown code fences if present. Returns an error dict rather than
    raising so that a batch run does not abort on a single bad response.
    """
    # Try to strip fences
    m = _JSON_FENCE_RE.search(raw)
    text = m.group(1) if m else raw.strip()

    # Find the outermost {...} block in case there is leading/trailing prose
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        return {"error": "parse_failed", "detail": str(exc), "raw": raw[:500]}

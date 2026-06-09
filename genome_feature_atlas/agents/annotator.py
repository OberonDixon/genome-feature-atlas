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

    def run(self, ctx: AgentContext, show_thinking: bool = False) -> dict:
        """Return a parsed annotation dict for the feature in ctx.

        If show_thinking is True, print the <think>...</think> block to stdout
        before returning so failure modes are visible during prompt iteration.
        """
        prompt = self._build_prompt(ctx)
        raw = self._client.ask(prompt)
        if show_thinking:
            thinking = _extract_thinking(raw)
            if thinking:
                print(f"[think] {thinking[:2000]}" + ("…" if len(thinking) > 2000 else ""))
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


_THINK_RE      = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_thinking(raw: str) -> str:
    m = _THINK_RE.search(raw)
    return m.group(1).strip() if m else ""


def _parse_json(raw: str) -> dict:
    """Extract and parse the JSON object from raw model output.

    If a complete <think>...</think> block is present, search for JSON only in
    the text after it, so brace characters in the thinking can't shadow the
    actual response. If thinking is present but truncated (no closing tag —
    model hit the token budget), fall back to searching for the LAST {...}
    block in the full text, which is more likely to be the JSON than any
    partial JSON-like structure inside the thinking.

    Returns an error dict rather than raising so batch runs don't abort.
    """
    close_idx = raw.rfind("</think>")
    if close_idx != -1:
        # Full thinking block present — search after it
        text = raw[close_idx + len("</think>"):].strip()
    elif "<think>" in raw:
        # Truncated thinking — search for the last {...} in the full text,
        # because the JSON (if any was generated) comes after the thinking
        text = raw.strip()
        # Use rfind("}") so we grab the last, not first, complete block
        end = text.rfind("}")
        if end == -1:
            return {"error": "parse_failed", "detail": "truncated thinking, no JSON", "raw": raw[:500]}
        start = text[:end+1].rfind("{")
        if start == -1 or start >= end:
            return {"error": "parse_failed", "detail": "no JSON block found", "raw": raw[:500]}
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError as exc:
            return {"error": "parse_failed", "detail": str(exc), "raw": raw[:500]}
    else:
        text = raw.strip()

    # Strip markdown fences if present
    m = _JSON_FENCE_RE.search(text)
    text = m.group(1) if m else text

    # Find the outermost {...} block in case there is leading/trailing prose
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        return {"error": "parse_failed", "detail": str(exc), "raw": raw[:500]}

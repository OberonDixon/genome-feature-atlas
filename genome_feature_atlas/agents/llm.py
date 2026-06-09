"""Thin wrapper around the OpenAI-compatible local vLLM API."""

from openai import OpenAI

_LOCAL_BASE_URL = "http://localhost:8000/v1"
_LOCAL_MODEL = "Qwen/Qwen3-8B"
# _LOCAL_MODEL = "Qwen/Qwen3-6-35B-A3B"


class LLMClient:
    def __init__(
        self,
        model=_LOCAL_MODEL,
        base_url=_LOCAL_BASE_URL,
        api_key="not-needed",
        thinking=False,
        system=None,
    ):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.thinking = thinking
        self.system = system

    def ask(self, prompt, thinking=None, max_tokens: int = 4096):
        """Send a prompt and return the response content as a string.

        prompt: str, or a list of message dicts for multi-turn.
        thinking: overrides instance default if provided.
        max_tokens: total token budget for thinking + response. 4096 is enough
            for a ~1500-token thinking block plus a ~200-token JSON response.
            Increase if thinking blocks are being truncated.
        """
        thinking = self.thinking if thinking is None else thinking

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = list(prompt)

        if self.system:
            messages = [{"role": "system", "content": self.system}] + messages

        extra_body = {}
        if not thinking:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        return response.choices[0].message.content

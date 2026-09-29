"""Thin wrapper around the OpenAI-compatible local vLLM API."""

from openai import OpenAI

_LOCAL_BASE_URL = "http://localhost:8000/v1"
# _LOCAL_MODEL = "Qwen/Qwen3-8B"
_LOCAL_MODEL = "Qwen/Qwen3.6-35B-A3B"


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

    # Match --max-model-len used when starting the vLLM server.
    # Update to 16384 after restarting with: --max-model-len 16384
    context_window: int = 8192

    def ask(self, prompt, thinking=None, max_tokens: int | None = None):
        """Send a prompt and return the response content as a string.

        prompt: str, or a list of message dicts for multi-turn.
        thinking: overrides instance default if provided.
        max_tokens: explicit output token budget. When None (default), the
            client reserves _OUTPUT_RESERVE tokens for output and uses the
            remaining context window. Raise _OUTPUT_RESERVE if thinking blocks
            are being truncated.
        """
        thinking = self.thinking if thinking is None else thinking

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = list(prompt)

        if self.system:
            messages = [{"role": "system", "content": self.system}] + messages

        if max_tokens is None:
            # BPE tokenizers encode roughly 2 chars per token for structured
            # prose/JSON (actual measurements: ~4500 tokens for ~9400 char prompt).
            # Subtract a 256-token safety margin so we never hit the server limit.
            prompt_chars = sum(len(m.get("content", "")) for m in messages)
            estimated_input = prompt_chars // 2
            max_tokens = max(512, self.context_window - estimated_input - 256)

        # Always pass enable_thinking explicitly so model behaviour is
        # deterministic regardless of server-side defaults.
        extra_body = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )
        return response.choices[0].message.content

"""OpenRouter backend.

Deliberately thin. The narrator does the validation, caching and fallback; this
only sends bytes. A backend that raises is handled -- the narrator degrades to a
template rather than failing the batch.

Model defaults to a free-tier one. Narration is off the matching path, batched
twenty at a time and cached by situation, so a 190,717-record batch costs a
handful of calls whatever the model.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"


class PaidModelRefused(RuntimeError):
    """Raised rather than silently spending credit on an optional feature."""


class OpenRouterBackend:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 timeout: float = 45.0, allow_paid: bool = False) -> None:
        """
        Args:
            allow_paid: permit a model that is not on the free tier. Off by
                default. Narration is an optional upgrade over templates that
                already work, so it should never quietly cost money -- and a
                loop over a large batch is exactly where an accidental paid
                model becomes expensive before anyone notices.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", _DEFAULT_MODEL)
        self.timeout = timeout
        
        self.last_latency_ms = 0.0
        self.last_prompt_chars = 0
        self.last_response_chars = 0
        self.total_calls = 0
        self.total_latency_ms = 0.0

        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        allow = allow_paid or os.environ.get("ALLOW_PAID_LLM", "").lower() in ("1", "true", "yes")
        if not self.model.endswith(":free") and not allow:
            raise PaidModelRefused(
                f"{self.model!r} is not a free-tier model. Narration falls back to "
                "templates at no cost. Set ALLOW_PAID_LLM=1 or pass allow_paid=True "
                "to spend credit deliberately."
            )

    def complete(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system",
                 "content": "You return only a JSON array. No prose, no code fences."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }).encode()

        req = urllib.request.Request(
            _ENDPOINT, data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())
        t1 = time.perf_counter()
        
        response_text = payload["choices"][0]["message"]["content"]
        
        self.last_latency_ms = (t1 - t0) * 1000.0
        self.last_prompt_chars = len(prompt)
        self.last_response_chars = len(response_text)
        self.total_calls += 1
        self.total_latency_ms += self.last_latency_ms
        
        return response_text

    @property
    def stats(self) -> dict[str, float | int | str]:
        return {
            "model": self.model,
            "last_latency_ms": self.last_latency_ms,
            "last_prompt_chars": self.last_prompt_chars,
            "last_response_chars": self.last_response_chars,
            "total_calls": self.total_calls,
            "total_latency_ms": self.total_latency_ms,
        }

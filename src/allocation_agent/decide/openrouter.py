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
import urllib.error
import urllib.request

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "google/gemini-2.0-flash-exp:free"


class OpenRouterBackend:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 timeout: float = 45.0) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", _DEFAULT_MODEL)
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

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
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())
        return payload["choices"][0]["message"]["content"]

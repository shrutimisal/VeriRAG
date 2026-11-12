"""Thin wrapper around the Google Gen AI SDK (`google-genai`) with retries and clean errors."""
from __future__ import annotations

import time

from src.utils.errors import LLMError, MissingAPIKeyError
from src.utils.logger import get_logger

log = get_logger(__name__)
_RETRYABLE_CODES = {429, 500, 502, 503, 504}


class GeminiClient:
    def __init__(self, api_key: str, model: str, temperature: float = 0.1, max_retries: int = 3):
        self.api_key = (api_key or "").strip()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if not self.api_key:
            raise MissingAPIKeyError(
                "GEMINI_API_KEY is not set. Add it to your .env file (see .env.example) "
                "or paste it into the sidebar."
            )
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        """Return the model's text. Retries on rate limits / transient server errors."""
        from google.genai import types

        client = self._get_client()
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.models.generate_content(model=self.model, contents=prompt, config=config)
                text = (response.text or "").strip()
                if not text:
                    raise LLMError("Gemini returned an empty response (it may have been blocked by safety filters).")
                return text
            except LLMError:
                raise
            except Exception as exc:  # SDK/network errors vary; classify by status code if present
                last_exc = exc
                code = getattr(exc, "code", None)
                retryable = code in _RETRYABLE_CODES or code is None  # None => likely network error
                if code in (400, 401, 403, 404):
                    raise LLMError(f"Gemini rejected the request (HTTP {code}): {exc}") from exc
                if not retryable or attempt == self.max_retries:
                    break
                wait = 2 ** attempt * 2
                log.warning("Gemini call failed (%s); retrying in %ss", exc, wait)
                time.sleep(wait)
        raise LLMError(f"Gemini request failed after retries: {last_exc}") from last_exc

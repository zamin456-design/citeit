"""LLM access behind a tiny interface, so Gemini can be swapped for another provider."""
import time
from typing import Protocol


class LLMError(Exception):
    """The language model service failed. The message is safe to show to the user."""


class RateLimitError(LLMError):
    """The free-tier rate limit was hit."""


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class GeminiClient:
    def __init__(self, api_key: str, model: str, temperature: float = 0.1, max_retries: int = 2):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set on the server. Add it to the .env file or the platform secrets.")
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        from google.genai import errors, types

        attempt = 0
        while True:
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=self.temperature),
                )
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                if code == 429:
                    if attempt >= self.max_retries:
                        raise RateLimitError(
                            "The free Gemini quota is temporarily used up. Please wait a minute and try again."
                        ) from exc
                    time.sleep(2 * (2**attempt))  # 2s, then 4s
                    attempt += 1
                    continue
                raise LLMError(f"The language model service returned an error (code {code}).") from exc
            except Exception as exc:  # network problems etc.
                raise LLMError("Could not reach the language model service.") from exc

            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise LLMError("The model returned an empty or blocked response.")
            return text

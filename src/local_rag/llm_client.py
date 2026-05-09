"""Ollama üzerinden yerel LLM bağlantısı.

Önkoşul:
    1. Ollama kurulu ve çalışıyor olmalı: https://ollama.com
    2. Bir model indirilmiş olmalı: `ollama pull llama3.2`
"""

from __future__ import annotations

from typing import Iterator

from .config import settings


class LLMClient:
    """Ollama'ya HTTP üzerinden istek atan basit istemci."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
    ) -> None:
        self.model = model or settings.llm_model
        self.host = host or settings.ollama_host
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.host)

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """Tek seferde tüm cevabı döndüren yardımcı."""
        self._ensure_client()
        assert self._client is not None

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature},
        )
        return response["message"]["content"]

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Token-by-token streaming generator."""
        self._ensure_client()
        assert self._client is not None

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for chunk in self._client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature},
            stream=True,
        ):
            yield chunk["message"]["content"]

"""Multi-provider LLM client for Forge.

Supports:
  - Ollama (local models) — default, no prefix or "ollama:" prefix
  - Gemini (Google cloud) — "gemini:" prefix, requires API key
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any


# ─── Provider helpers ─────────────────────────────────────────────────────────


def parse_provider(model: str) -> tuple[str, str]:
    """Split 'provider:model' into (provider, model_name).

    Returns ("ollama", model) for bare names.
    """
    if model.startswith("gemini:"):
        return "gemini", model[7:]
    if model.startswith("ollama:"):
        return "ollama", model[7:]
    # No prefix → Ollama
    return "ollama", model


def is_cloud_model(model: str) -> bool:
    """Return True if the model requires a cloud API (not local Ollama)."""
    provider, _ = parse_provider(model)
    return provider != "ollama"


# ─── Abstract backend ─────────────────────────────────────────────────────────


class _LLMBackend(ABC):
    """Common interface for LLM backends."""

    @abstractmethod
    def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> Generator[str, None, None]:
        ...

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


# ─── Ollama backend ──────────────────────────────────────────────────────────


class _OllamaBackend(_LLMBackend):
    """Local inference via Ollama."""

    def __init__(self, model: str, host: str) -> None:
        import ollama
        self.model = model
        self.client = ollama.Client(host=host)

    def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> Generator[str, None, None]:
        response = self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
        )
        for chunk in response:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,
        )
        return response["message"]["content"]


# ─── Gemini backend ──────────────────────────────────────────────────────────


class _GeminiBackend(_LLMBackend):
    """Google Gemini API backend."""

    def __init__(self, model: str, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                "Gemini API key not configured. "
                "Set it in ~/.config/forge/config.toml under [api_keys] "
                "or export GEMINI_API_KEY."
            )
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model_name = model
        self._genai = genai

    def _build_model(
        self, messages: list[dict[str, str]]
    ) -> tuple[Any, list[dict]]:
        """Create a GenerativeModel with system prompt, return (model, history).

        Converts from the Ollama/OpenAI message format:
            {"role": "system"|"user"|"assistant", "content": "..."}
        to Gemini's format:
            system_instruction for system messages
            {"role": "user"|"model", "parts": ["..."]} for history
        """
        system_parts: list[str] = []
        history: list[dict] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})

        model = self._genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction="\n\n".join(system_parts) if system_parts else None,
        )
        return model, history

    def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> Generator[str, None, None]:
        model, history = self._build_model(messages)

        # Gemini chat expects history without the last user message;
        # the last user message is sent via send_message().
        if history and history[-1]["role"] == "user":
            last_user = history.pop()
            prompt = last_user["parts"][0]
        else:
            prompt = ""

        chat = model.start_chat(history=history)
        response = chat.send_message(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def chat(self, messages: list[dict[str, str]]) -> str:
        model, history = self._build_model(messages)

        if history and history[-1]["role"] == "user":
            last_user = history.pop()
            prompt = last_user["parts"][0]
        else:
            prompt = ""

        chat = model.start_chat(history=history)
        response = chat.send_message(prompt)
        return response.text


# ─── Unified client ──────────────────────────────────────────────────────────


class LLMClient:
    """Unified LLM client — routes to the appropriate backend.

    Model naming:
        "deepseek-coder:33b"         → Ollama (local)
        "ollama:deepseek-coder:33b"  → Ollama (local, explicit)
        "gemini:gemini-2.0-flash"    → Google Gemini API
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        gemini_api_key: str = "",
    ) -> None:
        self.model = model
        provider, model_name = parse_provider(model)

        if provider == "gemini":
            self._backend: _LLMBackend = _GeminiBackend(model_name, gemini_api_key)
        else:
            self._backend = _OllamaBackend(model_name, host)

    def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> Generator[str, None, None]:
        """Stream a chat completion, yielding token chunks."""
        yield from self._backend.stream_chat(messages)

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Non-streaming chat completion."""
        return self._backend.chat(messages)

    def list_models(self) -> list[dict[str, Any]]:
        """List available models on the Ollama server."""
        if isinstance(self._backend, _OllamaBackend):
            response = self._backend.client.list()
            return response.get("models", [])
        return []

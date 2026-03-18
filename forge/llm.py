"""Multi-provider LLM client for Forge.

Supports:
  - Ollama (local models) — default, no prefix or "ollama:" prefix
  - Gemini (Google cloud) — "gemini:" prefix, requires API key
  - OpenAI — "openai:" prefix, requires API key
  - Anthropic — "anthropic:" prefix, requires API key
"""

from __future__ import annotations

import re as _re
import time as _time
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any


# ─── Stream chunk ─────────────────────────────────────────────────────────────


@dataclass
class StreamChunk:
    """A single chunk from a streaming LLM response."""
    text: str
    is_thinking: bool = False


# ─── Provider helpers ─────────────────────────────────────────────────────────


def parse_provider(model: str) -> tuple[str, str]:
    """Split 'provider:model' into (provider, model_name).

    Returns ("ollama", model) for bare names.
    """
    for prefix in ("gemini:", "openai:", "anthropic:", "ollama:"):
        if model.startswith(prefix):
            provider = prefix[:-1]
            return provider, model[len(prefix):]
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
    ) -> Generator[StreamChunk, None, None]:
        ...

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


# ─── Ollama backend ──────────────────────────────────────────────────────────


class _OllamaBackend(_LLMBackend):
    """Local inference via Ollama."""

    def __init__(
        self,
        model: str,
        host: str,
        temperature: float = 0.7,
        num_ctx: int = 16_384,
        num_predict: int = 4_096,
    ) -> None:
        import ollama
        self.model = model
        self.client = ollama.Client(host=host)
        self.options = {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }

    def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> Generator[StreamChunk, None, None]:
        response = self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options=self.options,
        )
        for chunk in response:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield StreamChunk(text=content)

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            options=self.options,
        )
        return response["message"]["content"]


# ─── Gemini backend ──────────────────────────────────────────────────────────


_RETRY_DELAY_RE = _re.compile(r"retry in ([\d.]+)s", _re.IGNORECASE)
_MAX_RETRIES = 3


def _parse_retry_delay(error: Exception) -> float:
    """Extract retry delay from a Gemini 429 error, default 60s."""
    m = _RETRY_DELAY_RE.search(str(error))
    return min(float(m.group(1)), 120.0) if m else 60.0


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is a 429 rate-limit error."""
    return "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error)


class _GeminiBackend(_LLMBackend):
    """Google Gemini API backend using the google-genai SDK."""

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        num_predict: int = 32_768,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Gemini API key not configured. "
                "Set it in ~/.config/forge/config.toml under [api_keys] "
                "or export GEMINI_API_KEY."
            )
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        self.temperature = temperature
        self.num_predict = num_predict

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict]]:
        """Convert from Ollama/OpenAI format to Gemini format.

        Returns (system_instruction, contents) where contents is the
        Gemini-format history list.
        """
        system_parts: list[str] = []
        contents: list[dict] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})

        system = "\n\n".join(system_parts) if system_parts else None
        return system, contents

    def _make_config(self, system: str | None):
        from google.genai import types
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=self.num_predict,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
            ),
        )

    def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> Generator[StreamChunk, None, None]:
        system, contents = self._convert_messages(messages)
        config = self._make_config(system)

        for attempt in range(_MAX_RETRIES + 1):
            try:
                for chunk in self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                ):
                    if not chunk.candidates:
                        continue
                    for part in chunk.candidates[0].content.parts:
                        if not part.text:
                            continue
                        yield StreamChunk(
                            text=part.text,
                            is_thinking=bool(getattr(part, "thought", False)),
                        )
                return  # success — exit retry loop
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < _MAX_RETRIES:
                    delay = _parse_retry_delay(e)
                    yield StreamChunk(
                        text=f"\n⏳ Rate limited — waiting {delay:.0f}s (attempt {attempt + 1}/{_MAX_RETRIES})...\n",
                        is_thinking=True,
                    )
                    _time.sleep(delay)
                else:
                    raise

    def chat(self, messages: list[dict[str, str]]) -> str:
        system, contents = self._convert_messages(messages)
        config = self._make_config(system)

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                parts = []
                for part in response.candidates[0].content.parts:
                    if part.text and not getattr(part, "thought", False):
                        parts.append(part.text)
                return "".join(parts)
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < _MAX_RETRIES:
                    delay = _parse_retry_delay(e)
                    _time.sleep(delay)
                else:
                    raise
        return ""  # unreachable, but satisfies type checker


# ─── OpenAI backend ──────────────────────────────────────────────────────────


class _OpenAIBackend(_LLMBackend):
    """OpenAI API backend (GPT-4o, GPT-4.1, etc.)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        num_predict: int = 4_096,
    ) -> None:
        if not api_key:
            raise ValueError(
                "OpenAI API key not configured. "
                "Set it in ~/.config/forge/config.toml under [api_keys] "
                "or export OPENAI_API_KEY."
            )
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model_name = model
        self.temperature = temperature
        self.num_predict = num_predict

    def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> Generator[StreamChunk, None, None]:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # already in OpenAI format
                    stream=True,
                    temperature=self.temperature,
                    max_tokens=self.num_predict,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        yield StreamChunk(text=delta.content)
                return
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < _MAX_RETRIES:
                    delay = _parse_retry_delay(e)
                    yield StreamChunk(
                        text=f"\n⏳ Rate limited — waiting {delay:.0f}s (attempt {attempt + 1}/{_MAX_RETRIES})...\n",
                        is_thinking=True,
                    )
                    _time.sleep(delay)
                else:
                    raise

    def chat(self, messages: list[dict[str, str]]) -> str:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.num_predict,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < _MAX_RETRIES:
                    delay = _parse_retry_delay(e)
                    _time.sleep(delay)
                else:
                    raise
        return ""


# ─── Anthropic backend ───────────────────────────────────────────────────────


class _AnthropicBackend(_LLMBackend):
    """Anthropic API backend (Claude Sonnet, Haiku, Opus)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        num_predict: int = 4_096,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Anthropic API key not configured. "
                "Set it in ~/.config/forge/config.toml under [api_keys] "
                "or export ANTHROPIC_API_KEY."
            )
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model
        self.temperature = temperature
        self.num_predict = num_predict

    def _split_system(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, list[dict[str, str]]]:
        """Extract system messages (Anthropic takes system as a separate param)."""
        system_parts: list[str] = []
        chat_msgs: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                chat_msgs.append(msg)
        return "\n\n".join(system_parts), chat_msgs

    def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> Generator[StreamChunk, None, None]:
        system, chat_msgs = self._split_system(messages)
        for attempt in range(_MAX_RETRIES + 1):
            try:
                with self.client.messages.stream(
                    model=self.model_name,
                    system=system,
                    messages=chat_msgs,
                    max_tokens=self.num_predict,
                    temperature=self.temperature,
                ) as stream:
                    for text in stream.text_stream:
                        yield StreamChunk(text=text)
                return
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < _MAX_RETRIES:
                    delay = _parse_retry_delay(e)
                    yield StreamChunk(
                        text=f"\n⏳ Rate limited — waiting {delay:.0f}s (attempt {attempt + 1}/{_MAX_RETRIES})...\n",
                        is_thinking=True,
                    )
                    _time.sleep(delay)
                else:
                    raise

    def chat(self, messages: list[dict[str, str]]) -> str:
        system, chat_msgs = self._split_system(messages)
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    system=system,
                    messages=chat_msgs,
                    max_tokens=self.num_predict,
                    temperature=self.temperature,
                )
                parts = []
                for block in response.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                return "".join(parts)
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < _MAX_RETRIES:
                    delay = _parse_retry_delay(e)
                    _time.sleep(delay)
                else:
                    raise
        return ""


# ─── Unified client ──────────────────────────────────────────────────────────


class LLMClient:
    """Unified LLM client — routes to the appropriate backend.

    Model naming:
        "deepseek-coder:33b"            → Ollama (local)
        "ollama:deepseek-coder:33b"     → Ollama (local, explicit)
        "gemini:gemini-2.5-flash"       → Google Gemini API
        "openai:gpt-4o"                 → OpenAI API
        "anthropic:claude-sonnet-4-..." → Anthropic API
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        gemini_api_key: str = "",
        openai_api_key: str = "",
        anthropic_api_key: str = "",
        temperature: float = 0.7,
        num_ctx: int = 16_384,
        num_predict: int = 4_096,
    ) -> None:
        self.model = model
        provider, model_name = parse_provider(model)

        if provider == "gemini":
            self._backend: _LLMBackend = _GeminiBackend(
                model_name,
                gemini_api_key,
                temperature=temperature,
                num_predict=max(num_predict, 32_768),
            )
        elif provider == "openai":
            self._backend = _OpenAIBackend(
                model_name,
                openai_api_key,
                temperature=temperature,
                num_predict=num_predict,
            )
        elif provider == "anthropic":
            self._backend = _AnthropicBackend(
                model_name,
                anthropic_api_key,
                temperature=temperature,
                num_predict=num_predict,
            )
        else:
            self._backend = _OllamaBackend(
                model_name,
                host,
                temperature=temperature,
                num_ctx=num_ctx,
                num_predict=num_predict,
            )

    def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> Generator[StreamChunk, None, None]:
        """Stream a chat completion, yielding StreamChunk objects."""
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

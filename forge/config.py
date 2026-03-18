"""Forge configuration with sensible defaults and optional TOML overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomli_w as _tomli_w
except ModuleNotFoundError:
    _tomli_w = None  # type: ignore[assignment]

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "forge" / "config.toml"

DEFAULT_MODEL = "deepseek-coder:33b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_OUTPUT = 10_000
DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_CTX = 16_384
DEFAULT_NUM_PREDICT = 4_096


@dataclass
class ForgeConfig:
    """Runtime configuration for Forge."""

    model: str = DEFAULT_MODEL
    ollama_host: str = DEFAULT_OLLAMA_HOST
    cwd: str = field(default_factory=os.getcwd)
    auto_approve: bool = False
    command_timeout: int = DEFAULT_TIMEOUT
    max_output_chars: int = DEFAULT_MAX_OUTPUT
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    temperature: float = DEFAULT_TEMPERATURE
    num_ctx: int = DEFAULT_NUM_CTX
    num_predict: int = DEFAULT_NUM_PREDICT

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        cli_model: str | None = None,
        cli_cwd: str | None = None,
    ) -> ForgeConfig:
        """Load config from TOML file, then override with CLI args."""
        data: dict = {}
        path = config_path or DEFAULT_CONFIG_PATH

        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)

        # API keys can be in [api_keys] section or at top level
        api_keys = data.get("api_keys", {})

        config = cls(
            model=data.get("model", DEFAULT_MODEL),
            ollama_host=data.get("ollama_host", DEFAULT_OLLAMA_HOST),
            cwd=data.get("cwd", os.getcwd()),
            auto_approve=data.get("auto_approve", False),
            command_timeout=data.get("command_timeout", DEFAULT_TIMEOUT),
            max_output_chars=data.get("max_output_chars", DEFAULT_MAX_OUTPUT),
            gemini_api_key=(
                api_keys.get("gemini", "")
                or data.get("gemini_api_key", "")
                or os.environ.get("GEMINI_API_KEY", "")
            ),
            openai_api_key=(
                api_keys.get("openai", "")
                or data.get("openai_api_key", "")
                or os.environ.get("OPENAI_API_KEY", "")
            ),
            anthropic_api_key=(
                api_keys.get("anthropic", "")
                or data.get("anthropic_api_key", "")
                or os.environ.get("ANTHROPIC_API_KEY", "")
            ),
            temperature=float(data.get("temperature", DEFAULT_TEMPERATURE)),
            num_ctx=int(data.get("num_ctx", DEFAULT_NUM_CTX)),
            num_predict=int(data.get("num_predict", DEFAULT_NUM_PREDICT)),
        )

        # CLI overrides take priority
        if cli_model:
            config.model = cli_model
        if cli_cwd:
            config.cwd = os.path.abspath(cli_cwd)

        return config

    def save(self, config_path: Path | None = None) -> None:
        """Persist current settings back to the TOML config file.

        Preserves existing keys (like api_keys) that we don't manage here.
        """
        path = config_path or DEFAULT_CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing data so we don't clobber keys we don't own
        existing: dict = {}
        if path.exists():
            with open(path, "rb") as f:
                existing = tomllib.load(f)

        # Update only the fields we manage
        existing["model"] = self.model
        existing["ollama_host"] = self.ollama_host
        existing["auto_approve"] = self.auto_approve
        existing["command_timeout"] = self.command_timeout
        existing["max_output_chars"] = self.max_output_chars
        existing["temperature"] = self.temperature
        existing["num_ctx"] = self.num_ctx
        existing["num_predict"] = self.num_predict

        if _tomli_w is not None:
            with open(path, "wb") as f:
                _tomli_w.dump(existing, f)
        else:
            # Minimal TOML writer — handles only the flat keys + [api_keys]
            with open(path, "w") as f:
                api_keys = existing.pop("api_keys", None)
                for k, v in existing.items():
                    if isinstance(v, bool):
                        f.write(f"{k} = {str(v).lower()}\n")
                    elif isinstance(v, float):
                        f.write(f"{k} = {v}\n")
                    elif isinstance(v, int):
                        f.write(f"{k} = {v}\n")
                    elif isinstance(v, str):
                        f.write(f'{k} = "{v}"\n')
                # Ensure current API keys are saved into the section
                if api_keys is None:
                    api_keys = {}
                if self.gemini_api_key:
                    api_keys["gemini"] = self.gemini_api_key
                if self.openai_api_key:
                    api_keys["openai"] = self.openai_api_key
                if self.anthropic_api_key:
                    api_keys["anthropic"] = self.anthropic_api_key
                if api_keys:
                    f.write("\n[api_keys]\n")
                    for k, v in api_keys.items():
                        f.write(f'{k} = "{v}"\n')

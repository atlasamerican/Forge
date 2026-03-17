"""Forge configuration with sensible defaults and optional TOML overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "forge" / "config.toml"

DEFAULT_MODEL = "deepseek-coder:33b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_OUTPUT = 10_000


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
        )

        # CLI overrides take priority
        if cli_model:
            config.model = cli_model
        if cli_cwd:
            config.cwd = os.path.abspath(cli_cwd)

        return config

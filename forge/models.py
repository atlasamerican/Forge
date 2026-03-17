"""Model catalog and management for Forge.

Curated categories of recommended models with descriptions,
plus utilities for pulling and updating models via Ollama.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class ModelEntry:
    name: str
    size: str       # Human-readable, e.g. "9 GB"
    vram_gb: float  # Approximate VRAM needed
    description: str


# ─── Curated Model Catalog ────────────────────────────────────────────────────

CATEGORIES: dict[str, dict] = {
    "coding": {
        "description": "Best models for writing, debugging, and reviewing code",
        "models": [
            ModelEntry("qwen2.5-coder:32b", "20 GB", 20.0,
                       "Top open-source coding model — excellent at complex refactoring and multi-file changes"),
            ModelEntry("qwen2.5-coder:14b", "9 GB", 9.0,
                       "Sweet spot of speed and quality — great for most coding tasks"),
            ModelEntry("deepseek-coder:33b", "18 GB", 18.0,
                       "Strong coding model — good at algorithms and data structures"),
            ModelEntry("codellama:34b", "19 GB", 19.0,
                       "Meta's code-focused Llama — solid Python, JS, C++ support"),
            ModelEntry("deepseek-coder-v2:16b", "9 GB", 9.0,
                       "Efficient MoE architecture — fast with good code quality"),
        ],
    },
    "linux": {
        "description": "Best for Linux system administration, shell scripting, and DevOps",
        "models": [
            ModelEntry("qwen2.5:32b", "20 GB", 20.0,
                       "Excellent at shell commands, systemd, networking, and package management"),
            ModelEntry("llama3.1:70b", "40 GB", 40.0,
                       "Most capable open model — needs RAM offload on 20GB GPU"),
            ModelEntry("qwen2.5:14b", "9 GB", 9.0,
                       "Great balance — strong at bash, Docker, and config files"),
            ModelEntry("llama3.1:8b", "5 GB", 5.0,
                       "Fast and light — good for simple shell commands and scripts"),
            ModelEntry("mistral:7b", "4 GB", 4.0,
                       "Efficient model with solid Linux and sysadmin knowledge"),
        ],
    },
    "general": {
        "description": "Well-rounded models for any task — planning, writing, analysis",
        "models": [
            ModelEntry("qwen2.5:32b", "20 GB", 20.0,
                       "Best all-around model for 20GB VRAM — strong reasoning"),
            ModelEntry("gemma2:27b", "16 GB", 16.0,
                       "Google's general-purpose model — good writing and analysis"),
            ModelEntry("llama3.1:8b", "5 GB", 5.0,
                       "Meta's versatile 8B — fast responses, good enough for many tasks"),
            ModelEntry("mistral:7b", "4 GB", 4.0,
                       "Lightweight and fast — great for quick Q&A and drafting"),
            ModelEntry("phi3:14b", "8 GB", 8.0,
                       "Microsoft's efficient model — strong at reasoning and math"),
        ],
    },
    "writing": {
        "description": "Best for documentation, README files, commit messages, and technical writing",
        "models": [
            ModelEntry("qwen2.5:32b", "20 GB", 20.0,
                       "Excellent prose quality and instruction following"),
            ModelEntry("llama3.1:8b", "5 GB", 5.0,
                       "Good writing quality at fast speed"),
            ModelEntry("gemma2:27b", "16 GB", 16.0,
                       "Strong at structured writing and documentation"),
            ModelEntry("mistral:7b", "4 GB", 4.0,
                       "Quick drafts and short-form content"),
            ModelEntry("phi3:14b", "8 GB", 8.0,
                       "Concise technical writing with good accuracy"),
        ],
    },
    "online": {
        "description": "Cloud API models — no local GPU needed, requires API key",
        "models": [
            ModelEntry("gemini:gemini-2.0-flash", "cloud", 0.0,
                       "Fast and capable — great for coding tasks, free tier available"),
            ModelEntry("gemini:gemini-2.0-flash-lite", "cloud", 0.0,
                       "Fastest Gemini model — lightweight tasks and quick responses"),
            ModelEntry("gemini:gemini-2.5-pro-preview-03-25", "cloud", 0.0,
                       "Most capable — complex reasoning, large context window"),
            ModelEntry("gemini:gemini-2.0-flash-thinking", "cloud", 0.0,
                       "Extended thinking for complex problems — shows reasoning steps"),
            ModelEntry("gemini:gemini-1.5-pro", "cloud", 0.0,
                       "Strong all-rounder with 1M token context window"),
        ],
    },
}


def get_installed_model_names() -> set[str]:
    """Get set of installed Ollama model names."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
        names = set()
        for line in result.stdout.strip().splitlines()[1:]:
            parts = line.split()
            if parts:
                names.add(parts[0])
        return names
    except (OSError, subprocess.TimeoutExpired):
        return set()


def format_category(
    category: str,
    gpu_vram_gb: float = 20.0,
    installed: set[str] | None = None,
) -> str:
    """Format a model category as Rich markup.

    Args:
        category: Category key (e.g. "coding", "linux").
        gpu_vram_gb: Available VRAM in GB (to mark models that fit).
        installed: Set of installed model names.
    """
    if installed is None:
        installed = get_installed_model_names()

    cat = CATEGORIES.get(category)
    if not cat:
        available = ", ".join(CATEGORIES.keys())
        return f"[bold red]Unknown category:[/] {category}\nAvailable: {available}"

    lines = []
    lines.append(f"[bold cyan]📦 {category.title()} Models[/] — {cat['description']}")
    lines.append("")

    for i, m in enumerate(cat["models"], 1):
        # Status indicators
        is_installed = m.name in installed
        fits_vram = m.vram_gb <= gpu_vram_gb

        status_parts = []
        if m.size == "cloud":
            status_parts.append("[bold blue]☁ cloud API[/]")
        elif is_installed:
            status_parts.append("[bold green]✓ installed[/]")
        if m.size != "cloud":
            if fits_vram:
                status_parts.append("[#888888]fits VRAM[/]")
            else:
                status_parts.append("[bold yellow]⚠ needs RAM offload[/]")

        status = " | ".join(status_parts)

        lines.append(f"  [bold]{i}. {m.name}[/] ({m.size})")
        lines.append(f"     {m.description}")
        lines.append(f"     {status}")
        lines.append("")

    if category == "online":
        lines.append("[dim]To switch:  /model <model_name>   (e.g. /model gemini:gemini-2.0-flash)[/]")
        lines.append("[dim]Requires API key in ~/.config/forge/config.toml[/]")
    else:
        lines.append("[dim]To install: /models pull <model_name>[/]")
        lines.append("[dim]To switch:  /model <model_name>[/]")

    return "\n".join(lines)


def format_all_categories(gpu_vram_gb: float = 20.0) -> str:
    """Format summary of all categories."""
    installed = get_installed_model_names()
    lines = []
    lines.append("[bold cyan]📦 Model Categories[/]")
    lines.append("")
    for key, cat in CATEGORIES.items():
        count_installed = sum(
            1 for m in cat["models"] if m.name in installed
        )
        lines.append(
            f"  [bold]{key}[/] — {cat['description']}"
            f" ({count_installed}/{len(cat['models'])} installed)"
        )
    lines.append("")
    lines.append("[dim]Usage: /models <category>   (e.g. /models coding)[/]")
    lines.append("[dim]       /models pull <name>  (download a model)[/]")
    lines.append("[dim]       /models update       (update all installed models)[/]")
    return "\n".join(lines)


def pull_model(name: str) -> subprocess.Popen:
    """Start pulling a model. Returns the Popen object for streaming output."""
    return subprocess.Popen(
        ["ollama", "pull", name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def update_all_models() -> list[str]:
    """Get list of installed model names for updating."""
    return sorted(get_installed_model_names())

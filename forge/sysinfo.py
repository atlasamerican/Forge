"""System hardware and environment info for Forge startup display."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GpuInfo:
    name: str
    vram_total_gb: float
    card_path: str  # e.g. /sys/class/drm/card1/device


@dataclass
class SystemInfo:
    cpu_model: str
    cpu_cores: int
    ram_total_gb: float
    gpu: GpuInfo | None
    ollama_models: list[dict[str, str]]  # [{"name": ..., "size": ...}, ...]


def get_cpu_info() -> tuple[str, int]:
    """Read CPU model and core count from /proc/cpuinfo."""
    model = "Unknown CPU"
    cores = 0
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name") and model == "Unknown CPU":
                    model = line.split(":", 1)[1].strip()
                if line.startswith("processor"):
                    cores += 1
    except OSError:
        pass
    return model, cores


def get_ram_total_gb() -> float:
    """Read total RAM from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / 1024 / 1024, 1)
    except OSError:
        pass
    return 0.0


def detect_gpu() -> GpuInfo | None:
    """Detect AMD GPU via sysfs (works without rocm-smi)."""
    drm_path = Path("/sys/class/drm")
    if not drm_path.exists():
        return None

    for card_dir in sorted(drm_path.iterdir()):
        device_dir = card_dir / "device"
        vram_total_file = device_dir / "mem_info_vram_total"

        if not vram_total_file.exists():
            continue

        # Read VRAM total
        try:
            vram_bytes = int(vram_total_file.read_text().strip())
            vram_gb = round(vram_bytes / (1024 ** 3), 1)
        except (OSError, ValueError):
            continue

        # Get GPU name from lspci
        gpu_name = _get_gpu_name_lspci()
        if not gpu_name:
            gpu_name = f"AMD GPU ({card_dir.name})"

        return GpuInfo(
            name=gpu_name,
            vram_total_gb=vram_gb,
            card_path=str(device_dir),
        )

    return None


def _get_gpu_name_lspci() -> str:
    """Get GPU name from lspci output."""
    try:
        result = subprocess.run(
            ["lspci"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            lower = line.lower()
            if ("vga" in lower or "3d" in lower or "display" in lower) and "amd" in lower:
                # Extract the part after the controller type
                parts = line.split(": ", 1)
                if len(parts) > 1:
                    name = parts[1].strip()
                    # Find all bracketed names — the last one is usually the specific model
                    import re
                    brackets = re.findall(r'\[([^\]]+)\]', name)
                    if len(brackets) >= 2:
                        return brackets[-1]  # e.g. "Radeon RX 7900 XT/7900 XTX/..."
                    elif brackets:
                        return brackets[0]
                    return name
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


def get_ollama_models() -> list[dict[str, str]]:
    """Get list of installed Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
        models = []
        for line in result.stdout.strip().splitlines()[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 3:
                models.append({
                    "name": parts[0],
                    "id": parts[1],
                    "size": parts[2] + " " + parts[3] if len(parts) > 3 else parts[2],
                })
        return models
    except (OSError, subprocess.TimeoutExpired):
        return []


def gather_system_info() -> SystemInfo:
    """Collect all system info."""
    cpu_model, cpu_cores = get_cpu_info()
    ram_gb = get_ram_total_gb()
    gpu = detect_gpu()
    models = get_ollama_models()

    return SystemInfo(
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        ram_total_gb=ram_gb,
        gpu=gpu,
        ollama_models=models,
    )


def format_startup_info(info: SystemInfo, active_model: str) -> str:
    """Format system info as Rich markup for the startup display."""
    lines = []
    lines.append("[bold cyan]╔══════════════════════════════════════════════════════╗[/]")
    lines.append("[bold cyan]║              ⚒  FORGE — AI Coding Terminal           ║[/]")
    lines.append("[bold cyan]╚══════════════════════════════════════════════════════╝[/]")
    lines.append("")

    # Hardware
    lines.append("[bold white]🖥  Hardware[/]")
    lines.append(f"   CPU: [bold]{info.cpu_model}[/] ({info.cpu_cores} threads)")
    lines.append(f"   RAM: [bold]{info.ram_total_gb} GB[/]")
    if info.gpu:
        lines.append(f"   GPU: [bold]{info.gpu.name}[/] ({info.gpu.vram_total_gb} GB VRAM)")
    else:
        lines.append("   GPU: [dim]Not detected (CPU inference only)[/]")
    lines.append("")

    # Models
    lines.append("[bold white]🤖  Installed Models[/]")
    if info.ollama_models:
        for m in info.ollama_models:
            marker = " ◀ active" if m["name"] == active_model else ""
            lines.append(
                f"   [bold]{m['name']}[/] ({m['size']})"
                f"{'[bold green]' + marker + '[/]' if marker else ''}"
            )
    else:
        lines.append("   [dim]No models found. Run: ollama pull deepseek-coder:33b[/]")
    lines.append("")

    # Tips
    lines.append("[bold white]⚡  Quick Start[/]")
    lines.append("   Type a request to get started, e.g.:")
    lines.append('   [italic]"Build me a Python script that converts CSV to JSON"[/]')
    lines.append("")
    lines.append("   [bold]/help[/] — commands  |  [bold]/auto[/] — toggle auto-approve  |  [bold]/models[/] — browse models")
    lines.append("")

    return "\n".join(lines)

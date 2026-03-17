"""Resource monitor for Forge — reads CPU, GPU, RAM stats.

Reads directly from /proc and sysfs for zero-dependency monitoring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResourceStats:
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    gpu_busy_percent: float | None   # None if no GPU detected
    vram_used_gb: float | None
    vram_total_gb: float | None
    vram_percent: float | None


class ResourceMonitor:
    """Polls system resources from /proc and sysfs."""

    def __init__(self) -> None:
        self._prev_cpu: tuple[float, float] | None = None  # (idle, total)
        self._gpu_card_path: str | None = self._find_gpu_card()

    def _find_gpu_card(self) -> str | None:
        """Find the sysfs path for the AMD GPU."""
        drm = Path("/sys/class/drm")
        if not drm.exists():
            return None
        for card_dir in sorted(drm.iterdir()):
            device = card_dir / "device"
            if (device / "gpu_busy_percent").exists():
                return str(device)
        return None

    def snapshot(self) -> ResourceStats:
        """Take a snapshot of current resource usage."""
        cpu = self._read_cpu()
        ram_used, ram_total, ram_pct = self._read_ram()
        gpu_busy = self._read_gpu_busy()
        vram_used, vram_total, vram_pct = self._read_vram()

        return ResourceStats(
            cpu_percent=cpu,
            ram_used_gb=ram_used,
            ram_total_gb=ram_total,
            ram_percent=ram_pct,
            gpu_busy_percent=gpu_busy,
            vram_used_gb=vram_used,
            vram_total_gb=vram_total,
            vram_percent=vram_pct,
        )

    def format_status_line(self, stats: ResourceStats) -> str:
        """Format stats as a compact string for the status bar."""
        parts = [
            f"CPU {stats.cpu_percent:4.1f}%",
            f"RAM {stats.ram_used_gb:.0f}/{stats.ram_total_gb:.0f}GB ({stats.ram_percent:.0f}%)",
        ]
        if stats.gpu_busy_percent is not None:
            parts.append(f"GPU {stats.gpu_busy_percent:.0f}%")
        if stats.vram_used_gb is not None and stats.vram_total_gb is not None:
            parts.append(
                f"VRAM {stats.vram_used_gb:.1f}/{stats.vram_total_gb:.1f}GB ({stats.vram_percent:.0f}%)"
            )
        return " | ".join(parts)

    def format_detailed(self, stats: ResourceStats) -> str:
        """Format stats as Rich markup for /stats display."""
        lines = [
            "[bold cyan]📊 System Resources[/]",
            "",
            f"  [bold]CPU:[/]  {stats.cpu_percent:5.1f}%  {_bar(stats.cpu_percent)}",
            f"  [bold]RAM:[/]  {stats.ram_used_gb:.1f} / {stats.ram_total_gb:.1f} GB"
            f"  ({stats.ram_percent:.1f}%)  {_bar(stats.ram_percent)}",
        ]
        if stats.gpu_busy_percent is not None:
            lines.append(
                f"  [bold]GPU:[/]  {stats.gpu_busy_percent:5.1f}%  {_bar(stats.gpu_busy_percent)}"
            )
        if stats.vram_used_gb is not None and stats.vram_total_gb is not None:
            lines.append(
                f"  [bold]VRAM:[/] {stats.vram_used_gb:.1f} / {stats.vram_total_gb:.1f} GB"
                f"  ({stats.vram_percent:.1f}%)  {_bar(stats.vram_percent)}"
            )
        return "\n".join(lines)

    # ─── Internal readers ─────────────────────────────────────────────────

    def _read_cpu(self) -> float:
        """Read CPU usage from /proc/stat (delta between two reads)."""
        try:
            with open("/proc/stat") as f:
                line = f.readline()
            parts = line.split()
            # user, nice, system, idle, iowait, irq, softirq, steal
            values = [float(v) for v in parts[1:9]]
            idle = values[3] + values[4]
            total = sum(values)

            if self._prev_cpu is None:
                self._prev_cpu = (idle, total)
                return 0.0

            prev_idle, prev_total = self._prev_cpu
            d_idle = idle - prev_idle
            d_total = total - prev_total
            self._prev_cpu = (idle, total)

            if d_total == 0:
                return 0.0
            return (1.0 - d_idle / d_total) * 100.0
        except (OSError, ValueError, IndexError):
            return 0.0

    def _read_ram(self) -> tuple[float, float, float]:
        """Read RAM from /proc/meminfo. Returns (used_gb, total_gb, percent)."""
        total_kb = avail_kb = 0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
                    if total_kb and avail_kb:
                        break
        except (OSError, ValueError):
            return 0.0, 0.0, 0.0

        total_gb = total_kb / 1024 / 1024
        used_gb = (total_kb - avail_kb) / 1024 / 1024
        pct = (used_gb / total_gb * 100) if total_gb > 0 else 0
        return round(used_gb, 1), round(total_gb, 1), round(pct, 1)

    def _read_gpu_busy(self) -> float | None:
        """Read GPU utilization from sysfs."""
        if not self._gpu_card_path:
            return None
        try:
            path = Path(self._gpu_card_path) / "gpu_busy_percent"
            return float(path.read_text().strip())
        except (OSError, ValueError):
            return None

    def _read_vram(self) -> tuple[float | None, float | None, float | None]:
        """Read VRAM usage from sysfs. Returns (used_gb, total_gb, percent)."""
        if not self._gpu_card_path:
            return None, None, None
        try:
            base = Path(self._gpu_card_path)
            total_bytes = int((base / "mem_info_vram_total").read_text().strip())
            used_bytes = int((base / "mem_info_vram_used").read_text().strip())
            total_gb = total_bytes / (1024 ** 3)
            used_gb = used_bytes / (1024 ** 3)
            pct = (used_gb / total_gb * 100) if total_gb > 0 else 0
            return round(used_gb, 1), round(total_gb, 1), round(pct, 1)
        except (OSError, ValueError):
            return None, None, None


def _bar(percent: float, width: int = 15) -> str:
    """Render a small text-based progress bar with Rich color."""
    filled = int(percent / 100 * width)
    empty = width - filled

    if percent >= 90:
        color = "#ff4444"
    elif percent >= 70:
        color = "#ffaa00"
    else:
        color = "#00ff88"

    return f"[{color}]{'█' * filled}[/][#444444]{'░' * empty}[/]"

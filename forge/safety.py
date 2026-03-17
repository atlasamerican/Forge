"""Command safety checker for Forge.

Three risk levels:
  - safe: benign commands (ls, cat, python, git, etc.)
  - dangerous: destructive commands that require user confirmation (rm, chmod, etc.)
  - blocked: catastrophically dangerous patterns (rm -rf /, fork bombs, etc.)

sudo on ANY command always requires confirmation regardless of auto-approve mode.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    SAFE = "safe"
    DANGEROUS = "dangerous"
    BLOCKED = "blocked"


@dataclass
class SafetyResult:
    level: RiskLevel
    reason: str


# Patterns that are blocked outright — never allowed
BLOCKED_PATTERNS: list[tuple[str, str]] = [
    (r"rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+.*)?/\s*$", "Recursive delete of root filesystem"),
    (r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+/\s*$", "Recursive forced delete of root"),
    (r"rm\s+-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+/\s*$", "Recursive forced delete of root"),
    (r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;?\s*:", "Fork bomb"),
    (r"dd\s+.*if=/dev/(zero|random|urandom)\s+.*of=/dev/[sh]d", "Disk overwrite with dd"),
    (r"mkfs\.", "Filesystem format command"),
    (r">\s*/dev/[sh]d[a-z]", "Direct write to block device"),
    (r"mv\s+.*/\s+/dev/null", "Moving root to /dev/null"),
]

# Commands that are always dangerous and require confirmation
DANGEROUS_COMMANDS: list[tuple[str, str]] = [
    (r"\brm\b", "File deletion (rm)"),
    (r"\bchmod\b", "Permission change (chmod)"),
    (r"\bchown\b", "Ownership change (chown)"),
    (r"\bkill\b", "Process kill"),
    (r"\bpkill\b", "Process kill by name"),
    (r"\bkillall\b", "Kill all processes by name"),
    (r"\bsystemctl\s+(stop|restart|disable|mask)", "Systemd service modification"),
    (r"\bpacman\s+-[A-Za-z]*[RS]", "Package manager removal/sync"),
    (r"\byay\s+-[A-Za-z]*[RS]", "AUR helper removal/sync"),
    (r"\bapt\s+(remove|purge|autoremove)", "Package removal"),
    (r"\bpip\s+install\b", "Python package installation"),
    (r"\bnpm\s+install\b", "Node package installation"),
    (r"\bcurl\b.*\|\s*(ba)?sh", "Pipe remote script to shell"),
    (r"\bwget\b.*\|\s*(ba)?sh", "Pipe remote script to shell"),
    (r"\breboot\b", "System reboot"),
    (r"\bshutdown\b", "System shutdown"),
    (r"\binit\s+[0-6]", "Runlevel change"),
]


def check_command(command: str) -> SafetyResult:
    """Evaluate a shell command and return its safety assessment.

    Args:
        command: The shell command string to evaluate.

    Returns:
        SafetyResult with risk level and reason.
    """
    stripped = command.strip()

    # sudo always requires confirmation
    if re.match(r"^\s*sudo\b", stripped):
        inner = re.sub(r"^\s*sudo\s+(-[A-Za-z]+\s+)*", "", stripped)
        inner_result = check_command(inner)
        if inner_result.level == RiskLevel.BLOCKED:
            return inner_result
        return SafetyResult(
            level=RiskLevel.DANGEROUS,
            reason=f"sudo: {inner_result.reason}" if inner_result.reason != "Safe command" else "Requires sudo privileges",
        )

    # Check blocked patterns first
    for pattern, reason in BLOCKED_PATTERNS:
        if re.search(pattern, stripped):
            return SafetyResult(level=RiskLevel.BLOCKED, reason=reason)

    # Check dangerous commands
    for pattern, reason in DANGEROUS_COMMANDS:
        if re.search(pattern, stripped):
            return SafetyResult(level=RiskLevel.DANGEROUS, reason=reason)

    return SafetyResult(level=RiskLevel.SAFE, reason="Safe command")

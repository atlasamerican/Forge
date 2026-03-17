"""Shell command execution manager for Forge.

Runs commands via subprocess with:
  - Virtual CWD tracking (handles `cd` commands)
  - Configurable timeout
  - Output truncation for large outputs
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass
class CommandResult:
    """Result of a shell command execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

    @property
    def output(self) -> str:
        """Combined stdout + stderr for display."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(self.stderr)
        return "\n".join(parts) if parts else "(no output)"

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class ShellManager:
    """Manages shell command execution with virtual CWD."""

    def __init__(
        self,
        cwd: str | None = None,
        timeout: int = 60,
        max_output: int = 10_000,
    ) -> None:
        self.cwd = os.path.abspath(cwd or os.getcwd())
        self.timeout = timeout
        self.max_output = max_output

    def run(self, command: str) -> CommandResult:
        """Execute a shell command and return the result.

        Handles `cd` commands by updating the virtual CWD.
        All other commands run via subprocess in the current virtual CWD.
        """
        stripped = command.strip()

        # Handle `cd` specially — update virtual CWD
        if stripped == "cd" or stripped.startswith("cd "):
            return self._handle_cd(stripped)

        try:
            result = subprocess.run(
                stripped,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.cwd,
                timeout=self.timeout,
                env={**os.environ, "TERM": "dumb"},
            )
            stdout = self._truncate(result.stdout)
            stderr = self._truncate(result.stderr)
            return CommandResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                stdout="",
                stderr=f"Command timed out after {self.timeout} seconds.",
                exit_code=-1,
                timed_out=True,
            )
        except Exception as e:
            return CommandResult(
                stdout="",
                stderr=f"Error executing command: {e}",
                exit_code=-1,
            )

    def _handle_cd(self, command: str) -> CommandResult:
        """Handle cd commands by updating virtual CWD."""
        if command.strip() == "cd":
            target = os.path.expanduser("~")
        else:
            target = command[3:].strip()
            # Remove surrounding quotes
            if (target.startswith('"') and target.endswith('"')) or (
                target.startswith("'") and target.endswith("'")
            ):
                target = target[1:-1]
            target = os.path.expanduser(target)

        # Resolve relative to current CWD
        if not os.path.isabs(target):
            target = os.path.join(self.cwd, target)
        target = os.path.realpath(target)

        if os.path.isdir(target):
            self.cwd = target
            return CommandResult(
                stdout=f"Changed directory to {self.cwd}",
                stderr="",
                exit_code=0,
            )
        else:
            return CommandResult(
                stdout="",
                stderr=f"cd: no such directory: {target}",
                exit_code=1,
            )

    def _truncate(self, text: str) -> str:
        """Truncate output that exceeds max_output chars."""
        if len(text) <= self.max_output:
            return text
        half = self.max_output // 2
        return (
            text[:half]
            + f"\n\n... [truncated {len(text) - self.max_output} characters] ...\n\n"
            + text[-half:]
        )

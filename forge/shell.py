"""Shell command execution manager for Forge.

Runs commands via subprocess with:
  - Virtual CWD tracking (handles `cd` commands)
  - Configurable timeout
  - Output truncation for large outputs
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections.abc import Callable
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

    def run_live(
        self,
        command: str,
        on_output: Callable[[str], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> CommandResult:
        """Execute a command with real-time streaming output.

        Uses Popen + a reader thread so each line of stdout/stderr is
        delivered to *on_output* as it arrives.  The main thread polls
        for completion, checking *is_cancelled* every 0.5 s.

        Timeout is **idle-based**: the process is only killed if it
        produces no output for *self.timeout* seconds.  As long as
        output keeps flowing the command runs indefinitely.
        """
        stripped = command.strip()

        # cd is handled in-process
        if stripped == "cd" or stripped.startswith("cd "):
            result = self._handle_cd(stripped)
            if on_output and result.stdout:
                on_output(result.stdout)
            return result

        env = {**os.environ, "TERM": "dumb", "PYTHONUNBUFFERED": "1"}

        try:
            proc = subprocess.Popen(
                stripped,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                cwd=self.cwd,
                env=env,
            )
        except Exception as e:
            return CommandResult(
                stdout="", stderr=f"Error executing command: {e}", exit_code=-1,
            )

        output_lines: list[str] = []
        # Shared mutable timestamp; updated by the reader thread each
        # time a line arrives so the main loop can detect idle hangs.
        last_activity = [time.monotonic()]

        def _reader() -> None:
            assert proc.stdout is not None
            try:
                for raw_line in proc.stdout:
                    line = raw_line.rstrip("\n")
                    output_lines.append(line)
                    last_activity[0] = time.monotonic()
                    if on_output:
                        on_output(line)
            except ValueError:
                pass  # stdout closed after kill

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        # Poll for completion — only timeout when idle (no output)
        timed_out = False
        cancelled = False
        while True:
            if is_cancelled and is_cancelled():
                proc.kill()
                cancelled = True
                break

            idle = time.monotonic() - last_activity[0]
            if idle > self.timeout:
                proc.kill()
                timed_out = True
                break

            try:
                proc.wait(timeout=0.5)
                break  # process finished normally
            except subprocess.TimeoutExpired:
                continue

        reader.join(timeout=5)

        full_output = "\n".join(output_lines)

        if cancelled:
            return CommandResult(
                stdout=self._truncate(full_output),
                stderr="Cancelled by user.",
                exit_code=-1,
            )
        if timed_out:
            return CommandResult(
                stdout=self._truncate(full_output),
                stderr=f"Command timed out (no output for {self.timeout}s).",
                exit_code=-1,
                timed_out=True,
            )

        return CommandResult(
            stdout=self._truncate(full_output),
            stderr="",
            exit_code=proc.returncode,
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

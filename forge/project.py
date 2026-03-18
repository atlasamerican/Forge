"""Per-project memory persistence for Forge.

Stores a compact project context (summary + recent actions) in a
`.forge-memory/` directory inside each project, giving the LLM
continuity across sessions.

Storage: <project_dir>/.forge-memory/context.json
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MEMORY_DIR_NAME = ".forge-memory"
MEMORY_FILE = "context.json"
MAX_ACTIONS = 50
TRIM_TO = 20  # after summarisation, keep this many recent actions

README_CONTENT = """# .forge-memory

This directory is managed by [Forge](https://github.com/atlasamerican/Forge),
a local AI coding assistant.

It stores project context (a summary and recent action log) so the AI
rememembers what has been done across sessions. You can safely delete
this directory to reset the AI's memory for this project.

Files:
- `context.json` — project name, summary, and action history
- `README.md`    — this file

This directory should be added to `.gitignore` (Forge offers to do this
automatically on setup).
"""


def _memory_dir(directory: str) -> Path:
    """Return the .forge-memory path for a project directory."""
    return Path(directory) / MEMORY_DIR_NAME


def _memory_file(directory: str) -> Path:
    """Return the context.json path for a project directory."""
    return _memory_dir(directory) / MEMORY_FILE


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _relative_age(iso_ts: str) -> str:
    """Return a human-readable age string like '2h ago' or '3d ago'."""
    try:
        then = datetime.fromisoformat(iso_ts)
        delta = datetime.now(timezone.utc) - then
        secs = int(delta.total_seconds())
        if secs < 60:
            return "just now"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return "unknown"


def _detect_project_name(directory: str) -> str:
    """Try to auto-detect a project name from the directory."""
    d = Path(directory)

    # Try pyproject.toml [project] name
    pyproject = d / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            name = data.get("project", {}).get("name")
            if name:
                return name
        except Exception:
            pass

    # Try package.json name
    pkg_json = d / "package.json"
    if pkg_json.exists():
        try:
            with open(pkg_json) as f:
                data = json.load(f)
            name = data.get("name")
            if name:
                return name
        except Exception:
            pass

    # Try git remote name
    try:
        result = subprocess.run(
            ["git", "-C", directory, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # Extract repo name from URL
            name = url.rstrip("/").rsplit("/", 1)[-1]
            if name.endswith(".git"):
                name = name[:-4]
            if name:
                return name
    except Exception:
        pass

    # Fallback: directory basename
    return d.name


@dataclass
class ActionEntry:
    """A single recorded action."""
    ts: str
    kind: str  # "task", "file", "cmd"
    text: str

    def to_dict(self) -> dict:
        return {"ts": self.ts, "kind": self.kind, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> ActionEntry:
        return cls(ts=d.get("ts", ""), kind=d.get("kind", ""), text=d.get("text", ""))


@dataclass
class ProjectMemory:
    """Persistent per-project memory."""

    directory: str = ""
    name: str = ""
    summary: str = ""
    actions: list[ActionEntry] = field(default_factory=list)
    last_active: str = ""
    last_model: str = ""

    @staticmethod
    def exists(directory: str) -> bool:
        """Return True if a .forge-memory/context.json exists for this directory."""
        abs_dir = str(Path(directory).resolve())
        return _memory_file(abs_dir).exists()

    @classmethod
    def load(cls, directory: str) -> ProjectMemory | None:
        """Load project memory for a directory, or return None if none exists.

        If the stored directory doesn't match (e.g. a copied project),
        the memory is re-homed: context is preserved but the directory
        is updated, the name is re-detected, and it is saved immediately.
        """
        abs_dir = str(Path(directory).resolve())
        path = _memory_file(abs_dir)

        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            stored_dir = data.get("directory", "")
            rehomed = bool(stored_dir and stored_dir != abs_dir)
            mem = cls(
                directory=abs_dir,
                name=data.get("name", ""),
                summary=data.get("summary", ""),
                actions=[
                    ActionEntry.from_dict(a)
                    for a in data.get("actions", [])
                ],
                last_active=data.get("last_active", ""),
                last_model=data.get("last_model", ""),
            )
            if rehomed:
                # Re-detect name for the new directory and persist
                mem.name = _detect_project_name(abs_dir)
                mem.save()
            return mem
        except Exception:
            return None

    @classmethod
    def create(cls, directory: str) -> ProjectMemory:
        """Create a new project memory directory and save initial context."""
        abs_dir = str(Path(directory).resolve())
        mem = cls(
            directory=abs_dir,
            name=_detect_project_name(abs_dir),
            last_active=_now_iso(),
        )
        mem.save()
        # Write the README
        readme_path = _memory_dir(abs_dir) / "README.md"
        if not readme_path.exists():
            readme_path.write_text(README_CONTENT)
        return mem

    @staticmethod
    def is_git_repo(directory: str) -> bool:
        """Return True if the directory is inside a git repository."""
        try:
            result = subprocess.run(
                ["git", "-C", directory, "rev-parse", "--git-dir"],
                capture_output=True, text=True, timeout=3,
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def needs_gitignore(directory: str) -> bool:
        """Return True if .forge-memory/ is not yet in .gitignore."""
        gitignore = Path(directory) / ".gitignore"
        if not gitignore.exists():
            return True
        content = gitignore.read_text()
        # Check for the entry (with or without trailing slash)
        for line in content.splitlines():
            stripped = line.strip()
            if stripped in (MEMORY_DIR_NAME, f"{MEMORY_DIR_NAME}/"):
                return False
        return True

    @staticmethod
    def add_to_gitignore(directory: str) -> None:
        """Append .forge-memory/ to the project's .gitignore."""
        gitignore = Path(directory) / ".gitignore"
        # Ensure we add on a new line
        if gitignore.exists():
            content = gitignore.read_text()
            if not content.endswith("\n"):
                content += "\n"
        else:
            content = ""
        content += f"# Forge AI project memory\n{MEMORY_DIR_NAME}/\n"
        gitignore.write_text(content)

    def save(self) -> None:
        """Persist project memory to disk."""
        self.last_active = _now_iso()
        mem_dir = _memory_dir(self.directory)
        mem_dir.mkdir(parents=True, exist_ok=True)
        path = mem_dir / MEMORY_FILE

        data = {
            "directory": self.directory,
            "name": self.name,
            "summary": self.summary,
            "actions": [a.to_dict() for a in self.actions],
            "last_active": self.last_active,
            "last_model": self.last_model,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def record_task(self, description: str) -> None:
        """Record a completed task (user's original request)."""
        self.actions.append(ActionEntry(
            ts=_now_iso(), kind="task", text=description,
        ))
        self._trim()

    def record_file(self, path: str, action: str = "modified") -> None:
        """Record a file operation."""
        self.actions.append(ActionEntry(
            ts=_now_iso(), kind="file", text=f"{action}: {path}",
        ))
        self._trim()

    def record_command(self, command: str) -> None:
        """Record a significant command execution."""
        # Skip very common/noisy commands
        skip_prefixes = ("ls", "cat ", "echo ", "pwd", "cd ")
        cmd_stripped = command.strip()
        if any(cmd_stripped.startswith(p) for p in skip_prefixes):
            return
        # Truncate very long commands
        if len(cmd_stripped) > 120:
            cmd_stripped = cmd_stripped[:117] + "..."
        self.actions.append(ActionEntry(
            ts=_now_iso(), kind="cmd", text=cmd_stripped,
        ))
        self._trim()

    def _trim(self) -> None:
        """Hard-trim to MAX_ACTIONS to prevent unbounded growth."""
        if len(self.actions) > MAX_ACTIONS:
            self.actions = self.actions[-MAX_ACTIONS:]

    def needs_summarization(self) -> bool:
        """Return True if the actions log is long enough to warrant compression."""
        return len(self.actions) >= MAX_ACTIONS

    def compress(self, new_summary: str) -> None:
        """Replace summary with a new one and trim actions."""
        self.summary = new_summary
        self.actions = self.actions[-TRIM_TO:]

    def format_for_prompt(self) -> str:
        """Format project context for injection into the system prompt."""
        if not self.summary and not self.actions:
            return ""

        lines = []
        lines.append(f"## Project Context")
        lines.append(f"Project: {self.name} ({self.directory})")

        if self.last_active:
            lines.append(f"Last session: {_relative_age(self.last_active)}")

        if self.summary:
            lines.append(f"Summary: {self.summary}")

        if self.actions:
            # Show the most recent actions (last 15 max to keep prompt compact)
            recent = self.actions[-15:]
            lines.append("Recent actions:")
            for a in recent:
                icon = {"task": "📋", "file": "📄", "cmd": "⚡"}.get(a.kind, "•")
                lines.append(f"  {icon} {a.text}")

        lines.append("")
        lines.append(
            "Use this context to understand the project state. "
            "Don't repeat work that was already done unless asked."
        )
        return "\n".join(lines)

    def format_display(self) -> str:
        """Format project context for user display (Rich markup)."""
        lines = []
        lines.append(f"[bold cyan]📁 Project:[/] {self.name}")
        lines.append(f"[bold]Directory:[/] {self.directory}")

        if self.last_active:
            lines.append(f"[bold]Last active:[/] {_relative_age(self.last_active)}")
        if self.last_model:
            lines.append(f"[bold]Last model:[/] {self.last_model}")

        if self.summary:
            lines.append(f"\n[bold]Summary:[/] {self.summary}")

        if self.actions:
            lines.append(f"\n[bold]Recent actions[/] ({len(self.actions)} total):")
            for a in self.actions[-10:]:
                icon = {"task": "📋", "file": "📄", "cmd": "⚡"}.get(a.kind, "•")
                lines.append(f"  {icon} [dim]{a.ts[:16]}[/] {a.text}")
        else:
            lines.append("\n[dim]No recorded actions yet.[/]")

        return "\n".join(lines)

    def clear(self) -> None:
        """Reset project memory."""
        self.summary = ""
        self.actions = []
        self.last_active = _now_iso()

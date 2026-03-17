"""Tool definitions and dispatcher for Forge agent.

Tools available to the LLM:
  - run_command: Execute a shell command
  - read_file: Read a file's contents
  - write_file: Write/create a file
  - list_directory: List directory contents
"""

from __future__ import annotations

import os
from typing import Any

from forge.shell import CommandResult, ShellManager

# Tool schemas — included in the system prompt so the LLM knows what's available
TOOL_SCHEMAS = """Available tools (use <tool_call> tags to invoke):

1. run_command — Execute a shell command
   {"name": "run_command", "arguments": {"command": "<shell command>"}}

2. read_file — Read the contents of a file
   {"name": "read_file", "arguments": {"path": "<file path>"}}

3. write_file — Write content to a file (creates or overwrites)
   {"name": "write_file", "arguments": {"path": "<file path>", "content": "<file content>"}}

4. list_directory — List files and directories
   {"name": "list_directory", "arguments": {"path": "<directory path>"}}"""


class ToolExecutor:
    """Executes tool calls from the agent."""

    def __init__(self, shell: ShellManager) -> None:
        self.shell = shell

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a tool call and return the result string."""
        handlers = {
            "run_command": self._run_command,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_directory": self._list_directory,
        }

        handler = handlers.get(name)
        if handler is None:
            return f"Error: Unknown tool '{name}'. Available: {', '.join(handlers)}"

        try:
            return handler(**arguments)
        except TypeError as e:
            return f"Error: Invalid arguments for '{name}': {e}"
        except Exception as e:
            return f"Error executing '{name}': {e}"

    def _run_command(self, command: str) -> str:
        """Execute a shell command. Returns combined output."""
        result: CommandResult = self.shell.run(command)
        status = "✓" if result.success else f"✗ (exit code {result.exit_code})"
        return f"[{status}]\n{result.output}"

    def _read_file(self, path: str) -> str:
        """Read a file's contents."""
        resolved = self._resolve_path(path)
        if not os.path.exists(resolved):
            return f"Error: File not found: {resolved}"
        if not os.path.isfile(resolved):
            return f"Error: Not a file: {resolved}"
        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if len(content) > 50_000:
                content = content[:50_000] + f"\n\n... [truncated, file is {len(content)} chars]"
            return content
        except Exception as e:
            return f"Error reading file: {e}"

    def _write_file(self, path: str, content: str) -> str:
        """Write content to a file, creating directories as needed."""
        resolved = self._resolve_path(path)
        try:
            os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {resolved}"
        except Exception as e:
            return f"Error writing file: {e}"

    def _list_directory(self, path: str = ".") -> str:
        """List directory contents."""
        resolved = self._resolve_path(path)
        if not os.path.exists(resolved):
            return f"Error: Directory not found: {resolved}"
        if not os.path.isdir(resolved):
            return f"Error: Not a directory: {resolved}"
        try:
            entries = sorted(os.listdir(resolved))
            lines = []
            for entry in entries:
                full = os.path.join(resolved, entry)
                prefix = "📁 " if os.path.isdir(full) else "📄 "
                lines.append(f"{prefix}{entry}")
            return "\n".join(lines) if lines else "(empty directory)"
        except Exception as e:
            return f"Error listing directory: {e}"

    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to the shell's CWD."""
        expanded = os.path.expanduser(path)
        if os.path.isabs(expanded):
            return expanded
        return os.path.join(self.shell.cwd, expanded)

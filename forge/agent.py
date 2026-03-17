"""Forge agent loop.

Orchestrates the LLM ↔ tool execution cycle:
  1. Send conversation to LLM (streaming)
  2. Parse response for <tool_call> tags
  3. Check safety, request approval if needed
  4. Execute tools, feed results back
  5. Loop until the LLM has no more tool calls
"""

from __future__ import annotations

import json
import platform
import re
import sys
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from forge.config import ForgeConfig
from forge.llm import LLMClient
from forge.safety import RiskLevel, SafetyResult, check_command
from forge.shell import ShellManager
from forge.tools import TOOL_SCHEMAS, ToolExecutor


class ApprovalAction(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    AUTO_APPROVE = "auto_approve"


@dataclass
class ToolCall:
    """A parsed tool call from the LLM response."""
    name: str
    arguments: dict[str, Any]
    raw: str  # The raw XML string


@dataclass
class AgentEvent:
    """An event emitted by the agent for the UI to display."""
    kind: str  # "text", "tool_call", "tool_result", "error", "thinking"
    content: str
    tool_call: ToolCall | None = None
    safety: SafetyResult | None = None


# Regex to find <tool_call>...</tool_call> blocks in LLM output
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)

# Fallback: find tool call JSON inside markdown code blocks (```json ... ```)
MARKDOWN_TOOL_PATTERN = re.compile(
    r"```(?:json)?\s*\n(.*?)\n\s*```",
    re.DOTALL | re.IGNORECASE,
)


def build_system_prompt(config: ForgeConfig, shell: ShellManager) -> str:
    """Build the system prompt with tool definitions and context."""
    return f"""You are Forge, a local AI coding assistant running in a terminal on the user's machine.
You help the user build, debug, and modify programs by writing code and running commands.

## Environment
- OS: {platform.system()} ({platform.freedesktop_os_release().get('NAME', 'Linux') if platform.system() == 'Linux' else platform.platform()})
- Python: {sys.version.split()[0]}
- Working directory: {shell.cwd}

## Tools
You have access to tools to interact with the user's system. To use a tool, output a <tool_call> tag containing valid JSON:

<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>

{TOOL_SCHEMAS}

## Rules
1. Think step by step. Plan your approach before writing code.
2. Use tools to accomplish tasks — write files, run commands, read existing code.
3. After writing code, RUN it to verify it works. If there are errors, read them and fix the code.
4. You can make multiple tool calls in sequence. After each tool result, decide your next action.
5. When you're done with a task, provide a brief summary of what you did.
6. If a command is rejected by the user, acknowledge it and find an alternative approach.
7. Keep your responses concise and focused on the task.
8. Use the working directory ({shell.cwd}) as the base for relative file paths.
9. When creating programs, include helpful comments and error handling.
10. IMPORTANT: Each response should contain EITHER text OR a single tool call, not both mixed together. If you need to explain something before using a tool, put the explanation first, then the tool call at the end.
11. CRITICAL: To create/modify files, you MUST use the write_file tool. Do NOT paste code as text.
12. CRITICAL: To create directories or install packages, you MUST use the run_command tool.
13. CRITICAL: Tool calls MUST use <tool_call> XML tags, NOT markdown code fences.
14. CRITICAL: NEVER say "I will use the run_command tool" and then show JSON in a markdown block. Instead, directly output the <tool_call> tag. The system only executes tool calls wrapped in <tool_call> tags.

## Correct tool call format
When you want to run a command, output EXACTLY this (no markdown, no explanation around it):

<tool_call>
{{"name": "run_command", "arguments": {{"command": "mkdir -p my_project"}}}}
</tool_call>

When you want to create a file:

<tool_call>
{{"name": "write_file", "arguments": {{"path": "my_project/main.py", "content": "print('hello')\n"}}}}
</tool_call>

## WRONG (will NOT execute):
```json
{{"name": "run_command", "arguments": {{"command": "mkdir -p my_project"}}}}
```
The above markdown format does NOT work. You MUST use <tool_call> tags."""


class Agent:
    """The Forge agent — drives the LLM ↔ tool loop."""

    def __init__(self, config: ForgeConfig) -> None:
        self.config = config
        self.llm = LLMClient(
            model=config.model,
            host=config.ollama_host,
            gemini_api_key=config.gemini_api_key,
        )
        self.shell = ShellManager(
            cwd=config.cwd,
            timeout=config.command_timeout,
            max_output=config.max_output_chars,
        )
        self.tools = ToolExecutor(self.shell)
        self.messages: list[dict[str, str]] = []
        self._cancelled = False

        # Initialize with system prompt
        self.messages.append({
            "role": "system",
            "content": build_system_prompt(config, self.shell),
        })
        self._stream_parse_buffer = ""

    def cancel(self) -> None:
        """Cancel the current agent operation."""
        self._cancelled = True

    def reset_cancel(self) -> None:
        """Reset the cancellation flag."""
        self._cancelled = False

    def clear_history(self) -> None:
        """Clear conversation history, keeping only the system prompt."""
        system = self.messages[0]
        self.messages = [system]

    def update_system_prompt(self) -> None:
        """Refresh the system prompt (e.g. after CWD change)."""
        self.messages[0] = {
            "role": "system",
            "content": build_system_prompt(self.config, self.shell),
        }

    def change_model(self, model: str) -> None:
        """Switch to a different model."""
        self.config.model = model
        self.llm = LLMClient(
            model=model,
            host=self.config.ollama_host,
            gemini_api_key=self.config.gemini_api_key,
        )

    def process_message(
        self,
        user_input: str,
        on_event: Callable[[AgentEvent], None],
        get_approval: Callable[[ToolCall, SafetyResult], ApprovalAction],
    ) -> None:
        """Process a user message through the agent loop.

        Args:
            user_input: The user's message.
            on_event: Callback for UI events (text chunks, tool calls, results).
            get_approval: Callback to get user approval for commands.
                          Must return an ApprovalAction.
        """
        self._cancelled = False
        self._stream_parse_buffer = ""
        self.messages.append({"role": "user", "content": user_input})

        # Agent loop — keep going until no more tool calls
        max_iterations = 25  # Safety limit
        for _ in range(max_iterations):
            if self._cancelled:
                on_event(AgentEvent(kind="text", content="\n\n*[Operation cancelled]*"))
                break

            # Get LLM response (streaming)
            full_response = ""
            try:
                for chunk in self.llm.stream_chat(self.messages):
                    if self._cancelled:
                        break
                    full_response += chunk
                    # Stream only content outside <tool_call> blocks.
                    # Tool calls are parsed/executed after full response collection.
                    visible = self._extract_visible_stream_text(chunk)
                    if visible:
                        on_event(AgentEvent(kind="text", content=visible))
            except Exception as e:
                on_event(AgentEvent(
                    kind="error",
                    content=f"LLM Error: {e}",
                ))
                break
            finally:
                # Flush any residual non-tool text in the stream buffer
                tail = self._flush_visible_stream_tail()
                if tail:
                    on_event(AgentEvent(kind="text", content=tail))

            # Add assistant response to history
            self.messages.append({"role": "assistant", "content": full_response})

            # Parse tool calls from response
            tool_calls = self._parse_tool_calls(full_response)

            if not tool_calls:
                # No tool calls — the agent is done responding
                break

            # Execute each tool call
            for tc in tool_calls:
                if self._cancelled:
                    break

                # For run_command, check safety
                if tc.name == "run_command":
                    command = tc.arguments.get("command", "")
                    safety = check_command(command)

                    if safety.level == RiskLevel.BLOCKED:
                        result_str = f"BLOCKED: {safety.reason}. This command cannot be executed."
                        on_event(AgentEvent(
                            kind="tool_result",
                            content=result_str,
                            tool_call=tc,
                            safety=safety,
                        ))
                        self._add_tool_result(tc, result_str, success=False)
                        continue

                    needs_approval = (
                        safety.level == RiskLevel.DANGEROUS
                        or not self.config.auto_approve
                    )

                    if needs_approval:
                        on_event(AgentEvent(
                            kind="tool_call",
                            content=command,
                            tool_call=tc,
                            safety=safety,
                        ))
                        action = get_approval(tc, safety)

                        if action == ApprovalAction.REJECT:
                            result_str = "Command was rejected by the user."
                            on_event(AgentEvent(
                                kind="tool_result",
                                content=result_str,
                                tool_call=tc,
                            ))
                            self._add_tool_result(tc, result_str, success=False)
                            continue
                        elif action == ApprovalAction.AUTO_APPROVE:
                            self.config.auto_approve = True
                    else:
                        # Auto-approved — just show what's running
                        on_event(AgentEvent(
                            kind="tool_call",
                            content=f"[auto] {command}",
                            tool_call=tc,
                            safety=safety,
                        ))
                else:
                    # Non-command tools (read_file, write_file, list_directory)
                    # are safe — auto-approve in auto mode, prompt in manual
                    if not self.config.auto_approve:
                        desc = f"{tc.name}: {json.dumps(tc.arguments)}"
                        safety = SafetyResult(level=RiskLevel.SAFE, reason="File operation")
                        on_event(AgentEvent(
                            kind="tool_call",
                            content=desc,
                            tool_call=tc,
                            safety=safety,
                        ))
                        action = get_approval(tc, safety)
                        if action == ApprovalAction.REJECT:
                            result_str = "Operation was rejected by the user."
                            self._add_tool_result(tc, result_str, success=False)
                            continue
                        elif action == ApprovalAction.AUTO_APPROVE:
                            self.config.auto_approve = True

                # Execute the tool
                result_str = self.tools.execute(tc.name, tc.arguments)
                success = not result_str.startswith("Error")

                on_event(AgentEvent(
                    kind="tool_result",
                    content=result_str,
                    tool_call=tc,
                ))
                self._add_tool_result(tc, result_str, success=success)

    def _parse_tool_calls(self, response: str) -> list[ToolCall]:
        """Extract tool calls from LLM response text."""
        calls = []

        # Primary: look for <tool_call>...</tool_call> XML tags
        for match in TOOL_CALL_PATTERN.finditer(response):
            raw_json = match.group(1).strip()
            # Strip accidental markdown fencing that some models emit
            if raw_json.startswith("```"):
                raw_json = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw_json)
                raw_json = re.sub(r"\n?```$", "", raw_json)
            tc = self._try_parse_tool_json(raw_json, match.group(0))
            if tc:
                calls.append(tc)

        # Fallback: if no XML tags worked, try markdown code blocks
        if not calls:
            for match in MARKDOWN_TOOL_PATTERN.finditer(response):
                raw_json = match.group(1).strip()
                tc = self._try_parse_tool_json(raw_json, match.group(0))
                if tc:
                    calls.append(tc)

        return calls

    @staticmethod
    def _try_parse_tool_json(raw_json: str, raw_match: str) -> ToolCall | None:
        """Try to parse a JSON string as a tool call. Returns None on failure."""
        for attempt in (raw_json, re.sub(r",(\s*[}\]])", r"\1", raw_json)):
            try:
                data = json.loads(attempt)
                if isinstance(data, dict) and "name" in data:
                    return ToolCall(
                        name=data["name"],
                        arguments=data.get("arguments", {}),
                        raw=raw_match,
                    )
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return None

    def _extract_visible_stream_text(self, chunk: str) -> str:
        """Return only text outside <tool_call>...</tool_call> tags from stream chunk."""
        self._stream_parse_buffer += chunk
        visible_parts: list[str] = []

        while self._stream_parse_buffer:
            start = self._stream_parse_buffer.find("<tool_call>")
            if start == -1:
                # Keep only a small suffix in case tag starts at chunk boundary
                keep = len("<tool_call>") - 1
                if len(self._stream_parse_buffer) > keep:
                    visible_parts.append(self._stream_parse_buffer[:-keep])
                    self._stream_parse_buffer = self._stream_parse_buffer[-keep:]
                break

            if start > 0:
                visible_parts.append(self._stream_parse_buffer[:start])
                self._stream_parse_buffer = self._stream_parse_buffer[start:]

            end = self._stream_parse_buffer.find("</tool_call>")
            if end == -1:
                # Wait for closing tag in later chunks
                break

            # Drop full tool_call block and continue scanning remainder
            self._stream_parse_buffer = self._stream_parse_buffer[end + len("</tool_call>"):]

        return "".join(visible_parts)

    def _flush_visible_stream_tail(self) -> str:
        """Flush remaining non-tool stream buffer at end of response."""
        if not self._stream_parse_buffer:
            return ""

        # If an incomplete tool_call is left, don't emit it to UI.
        if "<tool_call>" in self._stream_parse_buffer:
            prefix = self._stream_parse_buffer.split("<tool_call>", 1)[0]
            self._stream_parse_buffer = ""
            return prefix

        tail = self._stream_parse_buffer
        self._stream_parse_buffer = ""
        return tail

    def _add_tool_result(self, tc: ToolCall, result: str, success: bool) -> None:
        """Add a tool result to conversation history."""
        status = "true" if success else "false"
        self.messages.append({
            "role": "user",
            "content": f'<tool_result name="{tc.name}" success="{status}">\n{result}\n</tool_result>',
        })

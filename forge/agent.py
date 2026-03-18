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
from forge.instructions import InstructionManager
from forge.llm import LLMClient, StreamChunk
from forge.project import ProjectMemory
from forge.safety import RiskLevel, SafetyResult, check_command
from forge.shell import ShellManager
from forge.tools import TOOL_SCHEMAS, ToolExecutor


class ApprovalAction(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    AUTO_APPROVE = "auto_approve"
    CANCEL = "cancel"


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

# Qwen/ChatML models sometimes use <|im_start|> as delimiters
CHATML_TOOL_PATTERN = re.compile(
    r"<\|im_start\|>\s*(\{.*?\})\s*(?:<\|im_start\|>|<\|im_end\|>|$)",
    re.DOTALL,
)

# Fallback: find tool call JSON inside markdown code blocks (```json ... ```)
MARKDOWN_TOOL_PATTERN = re.compile(
    r"```(?:json)?\s*\n(.*?)\n\s*```",
    re.DOTALL | re.IGNORECASE,
)

# Known tool names for bare-JSON detection
_TOOL_NAMES = {"run_command", "write_file", "read_file", "list_directory"}


def build_system_prompt(
    config: ForgeConfig,
    shell: ShellManager,
    project: ProjectMemory | None = None,
    instruction_mgr: InstructionManager | None = None,
) -> str:
    """Build the system prompt with tool definitions and context."""
    project_context = ""
    if project:
        ctx = project.format_for_prompt()
        if ctx:
            project_context = f"\n\n{ctx}"

    instructions_section = ""
    references_section = ""
    if instruction_mgr:
        inst_text = instruction_mgr.format_instructions_for_prompt()
        if inst_text:
            instructions_section = f"\n\n{inst_text}"
        ref_text = instruction_mgr.format_references_for_prompt()
        if ref_text:
            references_section = f"\n\n{ref_text}"

    return f"""You are Forge, a local AI coding assistant running in a terminal on the user's machine.
You help the user build, debug, and modify programs by writing code and running commands.

## Environment
- OS: {platform.system()} ({platform.freedesktop_os_release().get('NAME', 'Linux') if platform.system() == 'Linux' else platform.platform()})
- Python: {sys.version.split()[0]}
- Working directory: {shell.cwd}
- Active model: {config.model}{project_context}{instructions_section}{references_section}

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
            openai_api_key=config.openai_api_key,
            anthropic_api_key=config.anthropic_api_key,
            temperature=config.temperature,
            num_ctx=config.num_ctx,
            num_predict=config.num_predict,
        )
        self.shell = ShellManager(
            cwd=config.cwd,
            timeout=config.command_timeout,
            max_output=config.max_output_chars,
        )
        self.tools = ToolExecutor(self.shell)
        self.messages: list[dict[str, str]] = []
        self._cancelled = False
        self.project: ProjectMemory | None = ProjectMemory.load(config.cwd)
        self._session_tool_calls: list[ToolCall] = []  # track calls per request

        # Load environment instructions
        proj_dir = self.project.directory if self.project else None
        self.instruction_mgr = InstructionManager(project_dir=proj_dir)

        # Initialize with system prompt
        self.messages.append({
            "role": "system",
            "content": build_system_prompt(config, self.shell, self.project, self.instruction_mgr),
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
            "content": build_system_prompt(self.config, self.shell, self.project, self.instruction_mgr),
        }

    def change_model(self, model: str) -> None:
        """Switch to a different model."""
        self.config.model = model
        self.llm = LLMClient(
            model=model,
            host=self.config.ollama_host,
            gemini_api_key=self.config.gemini_api_key,
            openai_api_key=self.config.openai_api_key,
            anthropic_api_key=self.config.anthropic_api_key,
            temperature=self.config.temperature,
            num_ctx=self.config.num_ctx,
            num_predict=self.config.num_predict,
        )
        self.update_system_prompt()

    def reload_llm_settings(self) -> None:
        """Recreate the LLM client with current config settings."""
        self.llm = LLMClient(
            model=self.config.model,
            host=self.config.ollama_host,
            gemini_api_key=self.config.gemini_api_key,
            openai_api_key=self.config.openai_api_key,
            anthropic_api_key=self.config.anthropic_api_key,
            temperature=self.config.temperature,
            num_ctx=self.config.num_ctx,
            num_predict=self.config.num_predict,
        )

    def load_project(self, directory: str | None = None) -> None:
        """Load project memory for a directory (None if no project file exists)."""
        d = directory or self.shell.cwd
        self.project = ProjectMemory.load(d)
        # Reload instructions for the new project directory
        proj_dir = self.project.directory if self.project else None
        self.instruction_mgr.set_project_dir(proj_dir)
        self.update_system_prompt()

    def init_project(self, directory: str | None = None) -> ProjectMemory:
        """Create a new project for a directory and activate it."""
        d = directory or self.shell.cwd
        self.project = ProjectMemory.create(d)
        self.update_system_prompt()
        return self.project

    def save_project(self) -> None:
        """Save the current project memory to disk (no-op if no active project)."""
        if self.project is None:
            return
        self.project.last_model = self.config.model
        self.project.save()

    def _record_to_project(self, user_input: str) -> None:
        """Record the completed request and tool calls to project memory."""
        if self.project is None or not self._session_tool_calls:
            return  # no active project or nothing substantive happened

        # Record the user's task (truncated)
        task_text = user_input[:200]
        if len(user_input) > 200:
            task_text += "..."
        self.project.record_task(task_text)

        # Record individual tool actions
        for tc in self._session_tool_calls:
            if tc.name == "run_command":
                cmd = tc.arguments.get("command", "")
                self.project.record_command(cmd)
            elif tc.name == "write_file":
                path = tc.arguments.get("path", "unknown")
                self.project.record_file(path, "wrote")
            elif tc.name == "read_file":
                pass  # reads aren't significant for project context
            elif tc.name == "list_directory":
                pass  # not significant

        # Update model and save
        self.project.last_model = self.config.model
        self.project.save()

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
        self._session_tool_calls = []
        self.messages.append({"role": "user", "content": user_input})

        # Agent loop — keep going until no more tool calls
        max_iterations = 25  # Safety limit
        for _ in range(max_iterations):
            if self._cancelled:
                on_event(AgentEvent(kind="text", content="\n\n*[Operation cancelled]*"))
                break

            # Get LLM response (streaming)
            full_response = ""
            was_thinking = False
            _rep_window = 200  # chars to compare for repetition
            _rep_count = 0
            _rep_max = 3  # stop after this many repeats
            try:
                for chunk in self.llm.stream_chat(self.messages):
                    if self._cancelled:
                        break
                    full_response += chunk.text

                    # ── Repetition guard ──────────────────────────
                    if not chunk.is_thinking and len(full_response) > _rep_window * 2:
                        tail = full_response[-_rep_window:]
                        prev = full_response[-_rep_window * 2:-_rep_window]
                        if tail == prev:
                            _rep_count += 1
                            if _rep_count >= _rep_max:
                                on_event(AgentEvent(
                                    kind="error",
                                    content="\n*[Stopped: model output was repeating]*",
                                ))
                                break
                        else:
                            _rep_count = 0

                    if chunk.is_thinking:
                        # Emit thinking events directly (no tool-call filtering)
                        on_event(AgentEvent(kind="thinking", content=chunk.text))
                        was_thinking = True
                    else:
                        if was_thinking:
                            # Transition from thinking → answer
                            on_event(AgentEvent(kind="thinking_done", content=""))
                            was_thinking = False
                        # Stream only content outside <tool_call> blocks.
                        visible = self._extract_visible_stream_text(chunk.text)
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
            tool_calls = self._parse_tool_calls(full_response, on_event=on_event)

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

                    # sudo always requires approval, even in auto mode
                    is_sudo = command.strip().startswith("sudo ")
                    needs_approval = is_sudo or not self.config.auto_approve

                    if needs_approval:
                        on_event(AgentEvent(
                            kind="tool_call",
                            content=command,
                            tool_call=tc,
                            safety=safety,
                        ))
                        action = get_approval(tc, safety)

                        if action == ApprovalAction.CANCEL:
                            result_str = "Operation cancelled by the user."
                            on_event(AgentEvent(
                                kind="tool_result",
                                content=result_str,
                                tool_call=tc,
                            ))
                            self._add_tool_result(tc, result_str, success=False)
                            self._cancelled = True
                            break
                        elif action == ApprovalAction.REJECT:
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
                        if action == ApprovalAction.CANCEL:
                            result_str = "Operation cancelled by the user."
                            self._add_tool_result(tc, result_str, success=False)
                            self._cancelled = True
                            break
                        elif action == ApprovalAction.REJECT:
                            result_str = "Operation was rejected by the user."
                            self._add_tool_result(tc, result_str, success=False)
                            continue
                        elif action == ApprovalAction.AUTO_APPROVE:
                            self.config.auto_approve = True

                # Execute the tool
                if tc.name == "run_command":
                    # Stream command output in real-time
                    cmd = tc.arguments.get("command", "")

                    def _on_line(line: str) -> None:
                        on_event(AgentEvent(kind="tool_output", content=line))

                    result = self.shell.run_live(
                        cmd,
                        on_output=_on_line,
                        is_cancelled=lambda: self._cancelled,
                    )

                    status = "\u2713" if result.success else f"\u2717 (exit code {result.exit_code})"
                    status_msg = f"[{status}]"
                    if result.timed_out or (not result.success and result.stderr):
                        status_msg += f"\n{result.stderr}"

                    on_event(AgentEvent(
                        kind="tool_result",
                        content=status_msg,
                        tool_call=tc,
                    ))
                    self._add_tool_result(
                        tc, f"[{status}]\n{result.output}", success=result.success,
                    )
                    self._session_tool_calls.append(tc)
                else:
                    result_str = self.tools.execute(tc.name, tc.arguments)
                    success = not result_str.startswith("Error")

                    on_event(AgentEvent(
                        kind="tool_result",
                        content=result_str,
                        tool_call=tc,
                    ))
                    self._add_tool_result(tc, result_str, success=success)
                    if success:
                        self._session_tool_calls.append(tc)

        # ── Record actions to project memory ──────────────────────────
        self._record_to_project(user_input)

    def _parse_tool_calls(
        self,
        response: str,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> list[ToolCall]:
        """Extract tool calls from LLM response text."""
        calls = []
        found_tags = 0

        # Primary: look for <tool_call>...</tool_call> XML tags
        for match in TOOL_CALL_PATTERN.finditer(response):
            found_tags += 1
            raw_json = match.group(1).strip()
            # Strip accidental markdown fencing that some models emit
            if raw_json.startswith("```"):
                raw_json = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw_json)
                raw_json = re.sub(r"\n?```$", "", raw_json)
            tc = self._try_parse_tool_json(raw_json, match.group(0))
            if tc:
                calls.append(tc)

        # Fallback 1: Qwen/ChatML <|im_start|> delimiters
        if not calls:
            for match in CHATML_TOOL_PATTERN.finditer(response):
                found_tags += 1
                raw_json = match.group(1).strip()
                tc = self._try_parse_tool_json(raw_json, match.group(0))
                if tc:
                    calls.append(tc)

        # Fallback 2: markdown code blocks (```json ... ```)
        if not calls:
            for match in MARKDOWN_TOOL_PATTERN.finditer(response):
                found_tags += 1
                raw_json = match.group(1).strip()
                tc = self._try_parse_tool_json(raw_json, match.group(0))
                if tc:
                    calls.append(tc)

        # Fallback 3: bare JSON with a known tool name (Qwen sometimes skips all tags)
        if not calls:
            for tc in self._scan_bare_json_tools(response):
                found_tags += 1
                calls.append(tc)

        # Warn if tool_call tags were found but none parsed successfully
        if found_tags > 0 and not calls and on_event:
            on_event(AgentEvent(
                kind="error",
                content=(
                    f"Found {found_tags} tool_call tag(s) but failed to parse "
                    "the JSON inside. The model may have produced malformed output. "
                    "Try rephrasing your request."
                ),
            ))

        return calls

    @staticmethod
    def _scan_bare_json_tools(response: str) -> list[ToolCall]:
        """Find bare JSON tool calls not wrapped in any tags.

        Scans each line for JSON objects that look like tool calls.
        Only matches known tool names to avoid false positives.
        """
        calls: list[ToolCall] = []
        for line in response.splitlines():
            stripped = line.strip()
            if not stripped.startswith("{") or not stripped.endswith("}"):
                continue
            try:
                data = json.loads(stripped)
                if (
                    isinstance(data, dict)
                    and data.get("name") in _TOOL_NAMES
                    and "arguments" in data
                ):
                    calls.append(ToolCall(
                        name=data["name"],
                        arguments=data["arguments"],
                        raw=stripped,
                    ))
            except (json.JSONDecodeError, TypeError):
                continue
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

    # Tags that delimit tool calls in the stream (should be hidden from UI)
    _TOOL_OPEN_TAGS = ("<tool_call>", "<|im_start|>")
    _TOOL_CLOSE_TAGS = {"<tool_call>": "</tool_call>", "<|im_start|>": "<|im_start|>"}

    def _extract_visible_stream_text(self, chunk: str) -> str:
        """Return only text outside tool-call delimiters from stream chunk.

        Supports both <tool_call>...</tool_call> and Qwen-style
        <|im_start|>...<|im_start|> blocks.
        """
        self._stream_parse_buffer += chunk
        visible_parts: list[str] = []

        while self._stream_parse_buffer:
            # Find the earliest opening tag
            best_start = -1
            best_tag = ""
            for tag in self._TOOL_OPEN_TAGS:
                idx = self._stream_parse_buffer.find(tag)
                if idx != -1 and (best_start == -1 or idx < best_start):
                    best_start = idx
                    best_tag = tag

            if best_start == -1:
                # No open tag found — keep a suffix in case one is arriving
                keep = max(len(t) for t in self._TOOL_OPEN_TAGS) - 1
                if len(self._stream_parse_buffer) > keep:
                    visible_parts.append(self._stream_parse_buffer[:-keep])
                    self._stream_parse_buffer = self._stream_parse_buffer[-keep:]
                break

            if best_start > 0:
                visible_parts.append(self._stream_parse_buffer[:best_start])
                self._stream_parse_buffer = self._stream_parse_buffer[best_start:]

            # Find the matching close tag
            close_tag = self._TOOL_CLOSE_TAGS[best_tag]
            # Search for close tag AFTER the open tag
            search_from = len(best_tag)
            end = self._stream_parse_buffer.find(close_tag, search_from)
            if end == -1:
                # Wait for closing tag in later chunks
                break

            # Drop the full tool block and continue
            self._stream_parse_buffer = self._stream_parse_buffer[end + len(close_tag):]

        return "".join(visible_parts)

    def _flush_visible_stream_tail(self) -> str:
        """Flush remaining non-tool stream buffer at end of response."""
        if not self._stream_parse_buffer:
            return ""

        # If an incomplete tool block is left, don't emit it to UI.
        for tag in self._TOOL_OPEN_TAGS:
            if tag in self._stream_parse_buffer:
                prefix = self._stream_parse_buffer.split(tag, 1)[0]
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

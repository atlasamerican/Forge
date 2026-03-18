"""Forge TUI — Textual-based terminal UI for the Forge AI coding assistant."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import threading

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.events import Key
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    Switch,
    TextArea,
)

from rich.markup import escape as rich_escape
from rich.text import Text

from forge.agent import Agent, AgentEvent, ApprovalAction, SafetyResult, ToolCall
from forge.config import ForgeConfig
from forge.models import format_all_categories, format_category, pull_model, update_all_models
from forge.monitor import ResourceMonitor, _bar
from forge.safety import RiskLevel
from forge.sysinfo import format_startup_info, gather_system_info, get_ollama_models


# ─── Approval Modal ──────────────────────────────────────────────────────────


class ApprovalModal(ModalScreen[ApprovalAction]):
    """Modal dialog for approving/rejecting a command."""

    BINDINGS = [
        Binding("y", "approve", "Approve"),
        Binding("n", "reject", "Reject"),
        Binding("a", "auto_approve", "Auto-approve all"),
        Binding("c", "cancel", "Cancel all"),
        Binding("escape", "reject", "Reject"),
    ]

    DEFAULT_CSS = """
    ApprovalModal {
        align: center middle;
    }

    #approval-dialog {
        width: 80;
        max-width: 90%;
        height: auto;
        max-height: 22;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    #approval-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }

    #approval-command {
        background: $boost;
        padding: 1;
        margin-bottom: 1;
        overflow-x: auto;
    }

    #approval-risk {
        color: $error;
        margin-bottom: 1;
    }

    #sudo-warning {
        color: $warning;
        text-style: bold;
        margin-bottom: 1;
    }

    #approval-buttons {
        height: 3;
        align: center middle;
    }

    #approval-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        command: str,
        safety: SafetyResult,
        is_command: bool = True,
        is_sudo: bool = False,
    ) -> None:
        super().__init__()
        self.command_text = command
        self.safety = safety
        self.is_command = is_command
        self.is_sudo = is_sudo

    def compose(self) -> ComposeResult:
        if self.is_sudo:
            title = "🔐 Sudo Command — Approval Required"
        elif self.is_command:
            title = "🔒 Command Approval Required"
        else:
            title = "📋 Operation Approval"
        risk_text = f"⚠ {self.safety.reason}" if self.safety.level == RiskLevel.DANGEROUS else ""

        with Vertical(id="approval-dialog"):
            yield Label(title, id="approval-title")
            if self.is_sudo:
                yield Label("⚠ Sudo commands always require manual approval.", id="sudo-warning")
            yield Static(self.command_text, id="approval-command")
            if risk_text:
                yield Label(risk_text, id="approval-risk")
            with Horizontal(id="approval-buttons"):
                yield Button("[Y] Approve", variant="success", id="btn-approve")
                yield Button("[N] Reject", variant="error", id="btn-reject")
                if not self.is_sudo:
                    yield Button("[A] Auto-approve", variant="warning", id="btn-auto")
                yield Button("[C] Cancel", variant="default", id="btn-cancel")

    def action_approve(self) -> None:
        self.dismiss(ApprovalAction.APPROVE)

    def action_reject(self) -> None:
        self.dismiss(ApprovalAction.REJECT)

    def action_auto_approve(self) -> None:
        if not self.is_sudo:
            self.dismiss(ApprovalAction.AUTO_APPROVE)

    def action_cancel(self) -> None:
        self.dismiss(ApprovalAction.CANCEL)

    @on(Button.Pressed, "#btn-approve")
    def on_approve(self) -> None:
        self.dismiss(ApprovalAction.APPROVE)

    @on(Button.Pressed, "#btn-reject")
    def on_reject(self) -> None:
        self.dismiss(ApprovalAction.REJECT)

    @on(Button.Pressed, "#btn-auto")
    def on_auto(self) -> None:
        self.dismiss(ApprovalAction.AUTO_APPROVE)

    @on(Button.Pressed, "#btn-cancel")
    def on_cancel(self) -> None:
        self.dismiss(ApprovalAction.CANCEL)


# ─── API Key Input Modal ──────────────────────────────────────────────────


class APIKeyModal(ModalScreen[str | None]):
    """Modal dialog to prompt the user for an API key."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    APIKeyModal {
        align: center middle;
    }

    #apikey-dialog {
        width: 70;
        max-width: 90%;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    #apikey-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }

    #apikey-hint {
        color: $text-muted;
        margin-bottom: 1;
    }

    #apikey-input {
        margin-bottom: 1;
    }

    #apikey-buttons {
        height: 3;
        align: center middle;
    }

    #apikey-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(self, provider: str) -> None:
        super().__init__()
        self.provider = provider

    def compose(self) -> ComposeResult:
        env_var = {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }.get(self.provider, "")
        with Vertical(id="apikey-dialog"):
            yield Label(f"🔑 API Key Required — {self.provider.title()}", id="apikey-title")
            yield Label(
                f"Paste your {self.provider.title()} API key below.\n"
                f"It will be saved to ~/.config/forge/config.toml\n"
                f"You can also set the {env_var} environment variable.",
                id="apikey-hint",
            )
            yield Input(placeholder="Paste API key here...", password=True, id="apikey-input")
            with Horizontal(id="apikey-buttons"):
                yield Button("Save", variant="success", id="btn-apikey-save")
                yield Button("Cancel", variant="default", id="btn-apikey-cancel")

    def on_mount(self) -> None:
        self.query_one("#apikey-input", Input).focus()

    @on(Input.Submitted, "#apikey-input")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        key = event.value.strip()
        self.dismiss(key if key else None)

    @on(Button.Pressed, "#btn-apikey-save")
    def on_save(self) -> None:
        key = self.query_one("#apikey-input", Input).value.strip()
        self.dismiss(key if key else None)

    @on(Button.Pressed, "#btn-apikey-cancel")
    def on_cancel_btn(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ─── Custom Messages ─────────────────────────────────────────────────────


class AgentEventMessage(Message):
    """Posted from the agent worker thread to the main thread."""

    def __init__(self, event: AgentEvent) -> None:
        super().__init__()
        self.event = event


class AgentDoneMessage(Message):
    """Posted when the agent finishes processing."""
    pass


class ApprovalRequestMessage(Message):
    """Posted when the agent needs user approval."""

    def __init__(
        self,
        tool_call: ToolCall,
        safety: SafetyResult,
        result_event: threading.Event,
    ) -> None:
        super().__init__()
        self.tool_call = tool_call
        self.safety = safety
        self.result_event = result_event
        self.action: ApprovalAction = ApprovalAction.REJECT


# ─── Tracking RichLog ─────────────────────────────────────────────────────────


class TrackingRichLog(RichLog):
    """RichLog that keeps a plain-text copy of everything written."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.plain_log: list[str] = []

    def write(self, content, *args, **kwargs):
        """Write content and also store a plain-text copy."""
        if isinstance(content, str):
            try:
                plain = Text.from_markup(content).plain
            except Exception:
                plain = content
        else:
            plain = str(content)
        self.plain_log.append(plain)
        return super().write(content, *args, **kwargs)


# ─── Wrapping Input Widget ───────────────────────────────────────────────────


class ChatInput(TextArea):
    """A TextArea configured as a wrapping chat input.

    Enter submits, Shift+Enter inserts a newline.
    Up/Down arrows navigate command history.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            language=None,
            soft_wrap=True,
            show_line_numbers=False,
            tab_behavior="focus",
            **kwargs,
        )
        self._history: list[str] = []
        self._history_index: int = -1  # -1 = new input (not browsing history)
        self._draft: str = ""  # saves in-progress text when browsing history

    class Submitted(Message):
        """Posted when the user presses Enter."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def _on_key(self, event: Key) -> None:
        if event.key == "enter":
            # Plain Enter submits; Shift+Enter is handled naturally as newline
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                # Add to history (avoid duplicating consecutive identical entries)
                if not self._history or self._history[-1] != text:
                    self._history.append(text)
                self._history_index = -1
                self._draft = ""
                self.post_message(self.Submitted(text))
                self.clear()
        elif event.key == "shift+enter":
            # Let Textual handle shift+enter as a normal newline
            pass
        elif event.key == "up":
            # Navigate backward through history
            if not self._history:
                return
            event.prevent_default()
            event.stop()
            if self._history_index == -1:
                # Save current draft before browsing
                self._draft = self.text
                self._history_index = len(self._history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            self._set_text(self._history[self._history_index])
        elif event.key == "down":
            # Navigate forward through history
            if self._history_index == -1:
                return
            event.prevent_default()
            event.stop()
            if self._history_index < len(self._history) - 1:
                self._history_index += 1
                self._set_text(self._history[self._history_index])
            else:
                # Back to the draft
                self._history_index = -1
                self._set_text(self._draft)

    def restore(self, text: str) -> None:
        """Restore text into the input (e.g. after a cancel)."""
        self._set_text(text)
        self._history_index = -1

    def _set_text(self, text: str) -> None:
        """Replace all content with the given text."""
        self.load_text(text)

    def clear(self) -> None:
        """Clear the text area content."""
        self.load_text("")


# ─── Main Application ────────────────────────────────────────────────────────


class ForgeApp(App):
    """The Forge AI terminal application."""

    TITLE = "Forge"
    SUB_TITLE = "Local AI Coding Assistant"

    BINDINGS = [
        Binding("ctrl+c", "cancel", "Cancel", show=True, priority=True),
        Binding("ctrl+q", "quit_app", "Quit", show=True),
        Binding("f1", "toggle_dashboard", "Dashboard", show=True),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    CSS = """
    #dashboard {
        height: auto;
        max-height: 40%;
        border: solid $accent;
        padding: 0 1;
        background: $surface;
    }

    #settings-row {
        height: auto;
        padding: 0;
        margin: 0;
    }

    #settings-row Label {
        padding: 1 1 0 0;
        width: auto;
        color: $text-muted;
    }

    #model-select {
        width: 1fr;
        min-width: 24;
        max-width: 40;
    }

    #temp-select {
        width: 12;
    }

    #ctx-select {
        width: 14;
    }

    #max-select {
        width: 14;
    }

    #auto-group {
        width: auto;
        height: auto;
        padding: 1 1 0 1;
    }

    #auto-group Label {
        padding: 0 1 0 0;
    }

    #project-btn {
        min-width: 16;
        max-width: 30;
        height: 3;
        margin: 0 0 0 1;
    }

    #resource-bars {
        height: auto;
        padding: 0 0;
        margin: 0;
    }

    #chat-log {
        height: 1fr;
        border: solid $primary;
        padding: 0 1;
        scrollbar-size: 1 1;
    }

    #input-bar {
        dock: bottom;
        height: auto;
        min-height: 3;
        max-height: 8;
        padding: 0 1;
    }

    #input-field {
        width: 1fr;
        min-height: 3;
        max-height: 6;
        border: solid $accent;
    }

    #active-prompt {
        height: auto;
        max-height: 3;
        padding: 0 1;
        background: $boost;
        color: $text-muted;
        display: none;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: $boost;
        padding: 0 1;
    }
    """

    auto_approve: reactive[bool] = reactive(False)
    is_processing: reactive[bool] = reactive(False)

    def __init__(self, config: ForgeConfig) -> None:
        super().__init__()
        self.config = config
        self.agent = Agent(config)
        self._pending_approval: ApprovalRequestMessage | None = None
        self._monitor = ResourceMonitor()
        self._sys_info = gather_system_info()
        self._last_input: str = ""  # last submitted text, for restore-on-cancel
        self._cached_models: list[dict] | None = None
        self._models_cache_tick: int = 0  # refresh model list every N dashboard updates
        self._clickable_commands: dict[int, str] = {}
        self._next_cmd_id: int = 0
        self._last_response_text: str = ""  # raw LLM text for /copy
        self._model_menu: list[str] = []  # numbered list from last /model display
        self._category_menu: list[str] = []  # model names from last /models <cat> display
        self._updating_widgets: bool = False  # guard against event loops

    # Preset options for settings dropdowns
    TEMP_OPTIONS = [(str(t), t) for t in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]]
    CTX_OPTIONS = [
        ("2k", 2048), ("4k", 4096), ("8k", 8192),
        ("16k", 16_384), ("32k", 32_768), ("64k", 65_536), ("128k", 131_072),
    ]
    MAX_TOKEN_OPTIONS = [
        ("1k", 1024), ("2k", 2048), ("4k", 4096),
        ("8k", 8192), ("16k", 16_384), ("32k", 32_768),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="dashboard"):
            with Horizontal(id="settings-row"):
                yield Label("Model")
                yield Select(
                    [(self.config.model, self.config.model)],
                    value=self.config.model,
                    id="model-select",
                    allow_blank=False,
                    compact=True,
                )
                yield Label("Temp")
                yield Select(
                    self.TEMP_OPTIONS,
                    value=self.config.temperature,
                    id="temp-select",
                    allow_blank=False,
                    compact=True,
                    tooltip=(
                        "Temperature — controls randomness\n"
                        "\n"
                        "Low (0.0-0.3): Focused, deterministic. Best for code generation.\n"
                        "Medium (0.5-0.7): Balanced creativity. Good default for most tasks.\n"
                        "High (1.0-2.0): Creative, varied. Good for brainstorming."
                    ),
                )
                yield Label("Ctx")
                yield Select(
                    self.CTX_OPTIONS,
                    value=self.config.num_ctx,
                    id="ctx-select",
                    allow_blank=False,
                    compact=True,
                    tooltip=(
                        "Context Window — how much text the model can \"see\"\n"
                        "\n"
                        "Includes your prompt, conversation history, and the response.\n"
                        "Larger = handles bigger files and longer conversations.\n"
                        "\n"
                        "8k: Simple tasks, quick questions\n"
                        "16k: Most coding tasks (recommended)\n"
                        "32k: Complex multi-file projects\n"
                        "64k+: Very large codebases (uses more VRAM)"
                    ),
                )
                yield Label("MaxOut")
                yield Select(
                    self.MAX_TOKEN_OPTIONS,
                    value=self.config.num_predict,
                    id="max-select",
                    allow_blank=False,
                    compact=True,
                    tooltip=(
                        "Max Output Tokens — limits response length\n"
                        "\n"
                        "Prevents runaway generation and controls cost.\n"
                        "\n"
                        "2k: Short answers, quick fixes\n"
                        "4k: Standard code generation (recommended)\n"
                        "8k-16k: Writing entire files or long explanations\n"
                        "32k: Very large outputs (may be slow on local models)"
                    ),
                )
                with Horizontal(id="auto-group"):
                    yield Label("Auto")
                    auto_sw = Switch(value=self.config.auto_approve, id="auto-switch")
                    auto_sw.tooltip = (
                        "Auto-Approve — skip confirmation for safe commands\n"
                        "\n"
                        "ON: Safe commands run immediately. Sudo still requires approval.\n"
                        "OFF: Every command asks for approval before running."
                    )
                    yield auto_sw
                proj_btn = Button(
                    self._project_button_label(),
                    id="project-btn",
                    variant="primary" if self.agent.project else "default",
                    tooltip=(
                        "Project Memory — persist context across sessions\n"
                        "\n"
                        "Click to create a project for the current directory,\n"
                        "or view/manage the active project."
                    ),
                )
                yield proj_btn
        yield Static("", id="resource-bars")
        yield Static("", id="active-prompt")
        yield TrackingRichLog(id="chat-log", wrap=True, highlight=True, markup=True)
        with Horizontal(id="input-bar"):
            yield ChatInput(id="input-field")
        yield Static(self._build_status(), id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app with system info and start resource monitor."""
        # Guard against Textual calling on_mount more than once
        if getattr(self, "_startup_done", False):
            return
        self._startup_done = True

        self.auto_approve = self.config.auto_approve
        log = self.query_one("#chat-log", RichLog)

        # Populate the model dropdown
        self._refresh_model_dropdown()

        # Show startup info
        startup_text = format_startup_info(self._sys_info, self.config.model)
        for line in startup_text.split("\n"):
            log.write(line)

        self.query_one("#input-field", ChatInput).focus()

        # Start resource monitor timer (updates every 2 seconds)
        self.set_interval(2.0, self._update_resources)
        self._update_resources()

    def _build_status(self) -> str:
        mode = "🟢 AUTO" if self.auto_approve else "🔴 MANUAL"
        model = self.config.model
        cwd = self.agent.shell.cwd
        if len(cwd) > 40:
            cwd = "..." + cwd[-37:]
        status = "⏳ Thinking..." if self.is_processing else "Ready"
        ctx_k = self.config.num_ctx // 1024
        return (f" {mode} | 🤖 {model} | ctx:{ctx_k}k "
                f"t:{self.config.temperature} max:{self.config.num_predict}"
                f" | 📁 {cwd} | {status}")

    def _update_resources(self) -> None:
        """Periodic update of the resource bars."""
        try:
            dashboard = self.query_one("#dashboard", Vertical)
            if dashboard.styles.display == "none":
                return  # skip work when hidden
            stats = self._monitor.snapshot()
            bars = self.query_one("#resource-bars", Static)
            bars.update(self._build_resource_bars(stats))
        except Exception:
            pass

    def _build_resource_bars(self, stats=None) -> str:
        """Build compact resource bar strings for the dashboard."""
        if stats is None:
            stats = self._monitor.snapshot()

        lines = []
        lines.append(f"  [bold]CPU[/] {stats.cpu_percent:5.1f}% {_bar(stats.cpu_percent, 12)}"
                     f"    [bold]RAM[/] {stats.ram_used_gb:.0f}/{stats.ram_total_gb:.0f}GB"
                     f" ({stats.ram_percent:.0f}%) {_bar(stats.ram_percent, 12)}")
        gpu_line = ""
        if stats.gpu_busy_percent is not None:
            gpu_line = f"  [bold]GPU[/] {stats.gpu_busy_percent:5.1f}% {_bar(stats.gpu_busy_percent, 12)}"
        if stats.vram_used_gb is not None and stats.vram_total_gb is not None:
            gpu_line += f"    [bold]VRAM[/] {stats.vram_used_gb:.1f}/{stats.vram_total_gb:.1f}GB"
            gpu_line += f" ({stats.vram_percent:.0f}%) {_bar(stats.vram_percent, 12)}"
        if gpu_line:
            lines.append(gpu_line)

        return "\n".join(lines)

    # ─── Project Button ──────────────────────────────────────────────────────

    def _project_button_label(self) -> str:
        """Return the label for the project button."""
        proj = self.agent.project
        if proj:
            name = proj.name
            if len(name) > 20:
                name = name[:18] + "…"
            return f"📁 {name}"
        return "📁 Init Project"

    def _refresh_project_button(self) -> None:
        """Update the project button label and variant."""
        try:
            btn = self.query_one("#project-btn", Button)
            btn.label = self._project_button_label()
            btn.variant = "primary" if self.agent.project else "default"
        except NoMatches:
            pass

    @on(Button.Pressed, "#project-btn")
    def on_project_btn_pressed(self) -> None:
        """Handle project button click."""
        log = self.query_one("#chat-log", RichLog)
        proj = self.agent.project
        if proj:
            # Show project info
            display = proj.format_display()
            for line in display.split("\n"):
                log.write(line)
        else:
            # Create project for current directory
            cwd = self.agent.shell.cwd
            new_proj = self.agent.init_project()
            log.write(f"[bold green]✓ Project created:[/] {new_proj.name}")
            log.write(f"[bold]Directory:[/] {new_proj.directory}")
            log.write(f"[dim]Memory stored in:[/] {cwd}/.forge-memory/")
            self._offer_gitignore(cwd, log)
            self._refresh_project_button()
            self._refresh_status()

    def _offer_gitignore(self, directory: str, log: RichLog) -> None:
        """If dir is a git repo and .forge-memory/ isn't gitignored, add it."""
        from forge.project import ProjectMemory
        if ProjectMemory.is_git_repo(directory) and ProjectMemory.needs_gitignore(directory):
            ProjectMemory.add_to_gitignore(directory)
            log.write("[bold green]✓[/] Added [bold].forge-memory/[/] to .gitignore")

    # ─── Model Dropdown ───────────────────────────────────────────────────────────

    def _get_api_key_for_provider(self, provider: str) -> str:
        """Return the configured API key for the given cloud provider."""
        return {
            "gemini": self.config.gemini_api_key,
            "openai": self.config.openai_api_key,
            "anthropic": self.config.anthropic_api_key,
        }.get(provider, "")

    def _set_api_key_for_provider(self, provider: str, key: str) -> None:
        """Store an API key for the given provider in config."""
        if provider == "gemini":
            self.config.gemini_api_key = key
        elif provider == "openai":
            self.config.openai_api_key = key
        elif provider == "anthropic":
            self.config.anthropic_api_key = key
        # Also propagate to agent config
        setattr(self.agent.config, f"{provider}_api_key", key)
        self.config.save()

    def _prompt_api_key(self, provider: str, model_name: str) -> None:
        """Show the API key modal and switch to model on success."""
        def on_key_result(key: str | None) -> None:
            if key:
                self._set_api_key_for_provider(provider, key)
                log = self.query_one("#chat-log", RichLog)
                log.write(f"[bold green]\u2713 {provider.title()} API key saved.[/]")
                # Now switch to the model
                self.agent.change_model(model_name)
                self.config.model = model_name
                self._cached_models = None
                self.config.save()
                self._refresh_model_dropdown()
                self._refresh_status()
                log.write(f"[bold green]\u2713 Switched to cloud model:[/] {model_name} [bold blue]\u2601[/]")
            else:
                # Cancelled — revert dropdown
                try:
                    self._updating_widgets = True
                    self.query_one("#model-select", Select).value = self.config.model
                finally:
                    self._updating_widgets = False

        self.push_screen(APIKeyModal(provider), callback=on_key_result)

    def _refresh_model_dropdown(self) -> None:
        """Rebuild the model Select options from Ollama + cloud models."""
        from forge.llm import is_cloud_model, parse_provider
        from forge.models import CATEGORIES, get_installed_model_names

        installed = sorted(get_installed_model_names())
        self._cached_models = get_ollama_models()

        all_models: list[str] = list(installed)
        # Add cloud models from the catalog that have a configured key
        for entry in CATEGORIES.get("online", {}).get("models", []):
            provider, _ = parse_provider(entry.name)
            if self._get_api_key_for_provider(provider):
                if entry.name not in all_models:
                    all_models.append(entry.name)

        self._model_menu = all_models

        # Build display labels
        options: list[tuple[str, str]] = []
        for name in all_models:
            label = f"☁ {name}" if is_cloud_model(name) else name
            options.append((label, name))

        try:
            self._updating_widgets = True
            sel = self.query_one("#model-select", Select)
            sel.set_options(options)
            if self.config.model in all_models:
                sel.value = self.config.model
        except NoMatches:
            pass
        finally:
            self._updating_widgets = False

    def watch_auto_approve(self, value: bool) -> None:
        self.config.auto_approve = value
        self.agent.config.auto_approve = value
        self.config.save()
        try:
            self.query_one("#status-bar", Static).update(self._build_status())
            # Keep switch in sync (e.g. when changed via /auto command)
            self._updating_widgets = True
            try:
                sw = self.query_one("#auto-switch", Switch)
                if sw.value != value:
                    sw.value = value
            finally:
                self._updating_widgets = False
        except NoMatches:
            pass

    def watch_is_processing(self, value: bool) -> None:
        try:
            self.query_one("#status-bar", Static).update(self._build_status())
            self.query_one("#input-field", ChatInput).disabled = value
        except NoMatches:
            pass

    # ─── Settings Event Handlers ──────────────────────────────────────────

    @on(Select.Changed, "#model-select")
    def on_model_select_changed(self, event: Select.Changed) -> None:
        """Handle model dropdown change."""
        if self._updating_widgets:
            return
        if event.value is Select.BLANK:
            return
        model_name = str(event.value)
        if model_name == self.config.model:
            return  # no change

        from forge.llm import is_cloud_model, parse_provider
        if is_cloud_model(model_name):
            provider, _ = parse_provider(model_name)
            if not self._get_api_key_for_provider(provider):
                self._prompt_api_key(provider, model_name)
                return

        self.agent.change_model(model_name)
        self.config.model = model_name
        self._cached_models = None
        self.config.save()
        self._refresh_status()
        log = self.query_one("#chat-log", RichLog)
        tag = " ☁" if is_cloud_model(model_name) else ""
        log.write(f"[bold green]✓ Switched to model:[/] {model_name}{tag}")

    @on(Select.Changed, "#temp-select")
    def on_temp_select_changed(self, event: Select.Changed) -> None:
        """Handle temperature dropdown change."""
        if self._updating_widgets:
            return
        if event.value is Select.BLANK:
            return
        val = float(event.value)
        if val == self.config.temperature:
            return
        self.config.temperature = val
        self.agent.config.temperature = val
        self.agent.reload_llm_settings()
        self.config.save()
        self._refresh_status()

    @on(Select.Changed, "#ctx-select")
    def on_ctx_select_changed(self, event: Select.Changed) -> None:
        """Handle context window dropdown change."""
        if self._updating_widgets:
            return
        if event.value is Select.BLANK:
            return
        val = int(event.value)
        if val == self.config.num_ctx:
            return
        self.config.num_ctx = val
        self.agent.config.num_ctx = val
        self.agent.reload_llm_settings()
        self.config.save()
        self._refresh_status()

    @on(Select.Changed, "#max-select")
    def on_max_select_changed(self, event: Select.Changed) -> None:
        """Handle max tokens dropdown change."""
        if self._updating_widgets:
            return
        if event.value is Select.BLANK:
            return
        val = int(event.value)
        if val == self.config.num_predict:
            return
        self.config.num_predict = val
        self.agent.config.num_predict = val
        self.agent.reload_llm_settings()
        self.config.save()
        self._refresh_status()

    @on(Switch.Changed, "#auto-switch")
    def on_auto_switch_changed(self, event: Switch.Changed) -> None:
        """Handle auto-approve switch toggle."""
        if self._updating_widgets:
            return
        self.auto_approve = event.value
        log = self.query_one("#chat-log", RichLog)
        state = "ON" if event.value else "OFF"
        log.write(f"[bold yellow]Auto-approve is now {state}[/]")

    # ─── Clickable Commands ──────────────────────────────────────────────

    def _make_clickable(self, cmd: str) -> str:
        """Register a command and return @click markup wrapping it."""
        cmd_id = self._next_cmd_id
        self._clickable_commands[cmd_id] = cmd
        self._next_cmd_id += 1
        return f"[@click=app.click_command({cmd_id})]{cmd}[/]"

    _BACKTICK_RE = re.compile(r'(?<!`)`([^`\n]+)`(?!`)')

    def _linkify_commands(self, text: str) -> str:
        """Replace `backtick` commands in text with clickable links."""
        def _replace(match: re.Match) -> str:
            cmd = match.group(1).strip()
            if not cmd or len(cmd) > 200:
                return match.group(0)
            cmd_id = self._next_cmd_id
            self._clickable_commands[cmd_id] = cmd
            self._next_cmd_id += 1
            return f"[@click=app.click_command({cmd_id})][bold cyan]▶ {cmd}[/][/]"
        return self._BACKTICK_RE.sub(_replace, text)

    def _copy_to_clipboard(self, text: str) -> bool:
        """Copy text to system clipboard with fallbacks."""
        try:
            self.copy_to_clipboard(text)
            return True
        except Exception:
            pass
        # Fallback: try system clipboard tools
        for tool in ["xclip -selection clipboard", "xsel --clipboard --input", "wl-copy"]:
            try:
                proc = subprocess.run(
                    tool.split(), input=text.encode(), capture_output=True, timeout=5
                )
                if proc.returncode == 0:
                    return True
            except Exception:
                continue
        return False

    # ─── Input Handling ───────────────────────────────────────────────────

    # Commands that are auto-detected as shell commands (never natural language).
    _SHELL_COMMANDS = {
        "cd", "ls", "pwd", "mkdir", "rmdir", "cp", "mv", "rm", "cat",
        "head", "tail", "touch", "chmod", "chown", "ln", "find", "grep",
        "wc", "sort", "uniq", "diff", "tar", "zip", "unzip",
        "git", "pip", "pip3", "npm", "yarn", "pnpm", "cargo", "make",
        "python", "python3", "node", "go", "rustc", "gcc", "g++",
        "docker", "kubectl", "curl", "wget", "ssh", "scp",
        "which", "whoami", "env", "export", "source", "echo",
    }

    @on(ChatInput.Submitted)
    def on_chat_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle user input from the chat input widget."""
        text = event.text.strip()
        if not text:
            return

        if text.startswith("/"):
            self._handle_slash_command(text)
            return

        # ! prefix — direct shell execution
        if text.startswith("!"):
            self._run_shell_direct(text[1:].strip())
            return

        # Auto-detect common shell commands
        first_word = text.split()[0] if text.split() else ""
        if first_word in self._SHELL_COMMANDS:
            self._run_shell_direct(text)
            return

        # Remember this input so we can restore it on cancel
        self._last_input = text
        self._last_response_text = ""  # reset for new response

        log = self.query_one("#chat-log", RichLog)
        log.write(f"\n[bold #00aaff]You:[/] {text}")

        self.is_processing = True
        # Show the active prompt below the dashboard
        try:
            prompt_display = self.query_one("#active-prompt", Static)
            truncated = text if len(text) <= 120 else text[:117] + "..."
            prompt_display.update(f"💬 {truncated}")
            prompt_display.styles.display = "block"
        except NoMatches:
            pass
        self._run_agent(text)

    def _run_shell_direct(self, command: str) -> None:
        """Run a shell command directly without the agent."""
        log = self.query_one("#chat-log", RichLog)
        if not command:
            log.write("[bold red]Usage:[/] !<command>  (e.g. !ls, !git status)")
            return

        log.write(f"\n[bold #ffaa00]$ {command}[/]")
        result = self.agent.shell.run(command)

        # cd changes directory — update project/status like /cwd does
        stripped = command.strip()
        if stripped == "cd" or stripped.startswith("cd "):
            if result.success:
                new_path = self.agent.shell.cwd
                self.agent.save_project()
                self.agent.load_project(new_path)
                self.agent.update_system_prompt()
                proj = self.agent.project
                if proj:
                    log.write(f"[bold cyan]📁 Project:[/] {proj.name}")
                self._refresh_project_button()
                self._refresh_status()

        # Display output
        output = result.output
        if output and output != "(no output)":
            # Truncate display for very long output
            if len(output) > 3000:
                output = output[:3000] + "\n... [truncated for display]"
            log.write(f"[#888888]{output}[/]")

        if not result.success and result.stderr:
            log.write(f"[bold red]{result.stderr}[/]")
        elif not result.success:
            log.write(f"[dim]Exit code: {result.exit_code}[/]")

    def _handle_slash_command(self, text: str) -> None:
        """Handle slash commands."""
        log = self.query_one("#chat-log", RichLog)
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/auto":
            self.auto_approve = not self.auto_approve
            state = "ON" if self.auto_approve else "OFF"
            log.write(f"[bold yellow]Auto-approve is now {state}[/]")

        elif cmd == "/model":
            self._handle_model_command(arg, log)

        elif cmd == "/models":
            self._handle_models_command(arg, log)

        elif cmd == "/clear":
            self.agent.clear_history()
            log.clear()
            log.write("[bold cyan]Chat history cleared.[/]")

        elif cmd == "/cwd":
            if not arg:
                log.write(f"[bold]Current directory:[/] {self.agent.shell.cwd}")
                log.write("Usage: /cwd <path>")
            else:
                path = os.path.abspath(os.path.expanduser(arg))
                if os.path.isdir(path):
                    # Save current project before switching
                    self.agent.save_project()
                    self.agent.shell.cwd = path
                    # Load project if one exists for this directory
                    self.agent.load_project(path)
                    log.write(f"[bold green]Working directory:[/] {path}")
                    proj = self.agent.project
                    if proj:
                        log.write(f"[bold cyan]📁 Project:[/] {proj.name}")
                        if proj.summary:
                            log.write(f"[dim]{proj.summary}[/]")
                    else:
                        log.write("[dim]No project file here. Use /project init to create one.[/]")
                    self._refresh_project_button()
                    self._refresh_status()
                else:
                    log.write(f"[bold red]Not a directory:[/] {path}")

        elif cmd in ("/exit", "/quit"):
            self.exit()
            return

        elif cmd in ("/dashboard", "/panel"):
            self.action_toggle_dashboard()

        elif cmd == "/copy":
            if self._last_response_text.strip():
                if self._copy_to_clipboard(self._last_response_text.strip()):
                    log.write("[bold green]Copied last response to clipboard.[/]")
                else:
                    log.write("[bold red]Clipboard not available.[/] "
                              "Install xclip, xsel, or wl-copy.")
            else:
                log.write("[bold yellow]Nothing to copy.[/]")

        elif cmd == "/savelog":
            tracking_log = self.query_one("#chat-log", TrackingRichLog)
            lines = tracking_log.plain_log
            log_path = os.path.expanduser("~/.config/forge/session.log")
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                log.write(f"[bold green]Saved {len(lines)} lines to:[/] {log_path}")
            except Exception as e:
                log.write(f"[bold red]Error saving log:[/] {e}")

        elif cmd == "/copylog":
            tracking_log = self.query_one("#chat-log", TrackingRichLog)
            if arg.isdigit():
                lines = tracking_log.plain_log[-int(arg):]
            else:
                lines = tracking_log.plain_log
            full_text = "\n".join(lines)
            if self._copy_to_clipboard(full_text):
                count = len(lines)
                log.write(f"[bold green]Copied {count} log lines to clipboard.[/]")
            else:
                log.write("[bold red]Clipboard not available.[/] "
                          "Install xclip, xsel, or wl-copy.")

        elif cmd == "/stats":
            stats = self._monitor.snapshot()
            detailed = self._monitor.format_detailed(stats)
            for line in detailed.split("\n"):
                log.write(line)

        elif cmd == "/temp":
            if not arg:
                log.write(f"[bold]Temperature:[/] {self.config.temperature}")
                log.write("Usage: /temp <0.0-2.0>  (lower = more focused, higher = more creative)")
            else:
                try:
                    val = float(arg)
                    if not 0.0 <= val <= 2.0:
                        log.write("[bold red]Temperature must be between 0.0 and 2.0[/]")
                    else:
                        self.config.temperature = val
                        self.agent.config.temperature = val
                        self.agent.reload_llm_settings()
                        self.config.save()
                        self._refresh_status()
                        self._sync_select("#temp-select", val)
                        log.write(f"[bold green]✓ Temperature set to {val}[/]")
                except ValueError:
                    log.write("[bold red]Invalid number.[/] Usage: /temp <0.0-2.0>")

        elif cmd == "/ctx":
            if not arg:
                log.write(f"[bold]Context window:[/] {self.config.num_ctx:,} tokens")
                log.write("Usage: /ctx <tokens>  (e.g. /ctx 8192, /ctx 32768)")
            else:
                try:
                    val = int(arg.replace("k", "000").replace("K", "000"))
                    if val < 1024:
                        log.write("[bold red]Context must be at least 1024 tokens[/]")
                    elif val > 131_072:
                        log.write("[bold red]Context must be at most 131072 tokens[/]")
                    else:
                        self.config.num_ctx = val
                        self.agent.config.num_ctx = val
                        self.agent.reload_llm_settings()
                        self.config.save()
                        self._refresh_status()
                        self._sync_select("#ctx-select", val)
                        log.write(f"[bold green]✓ Context window set to {val:,} tokens[/]")
                except ValueError:
                    log.write("[bold red]Invalid number.[/] Usage: /ctx <tokens>")

        elif cmd == "/maxtokens":
            if not arg:
                log.write(f"[bold]Max output tokens:[/] {self.config.num_predict:,}")
                log.write("Usage: /maxtokens <tokens>  (e.g. /maxtokens 4096, /maxtokens 8192)")
            else:
                try:
                    val = int(arg.replace("k", "000").replace("K", "000"))
                    if val < 256:
                        log.write("[bold red]Must be at least 256 tokens[/]")
                    elif val > 65_536:
                        log.write("[bold red]Must be at most 65536 tokens[/]")
                    else:
                        self.config.num_predict = val
                        self.agent.config.num_predict = val
                        self.agent.reload_llm_settings()
                        self.config.save()
                        self._refresh_status()
                        self._sync_select("#max-select", val)
                        log.write(f"[bold green]✓ Max output tokens set to {val:,}[/]")
                except ValueError:
                    log.write("[bold red]Invalid number.[/] Usage: /maxtokens <tokens>")

        elif cmd == "/project":
            self._handle_project_command(arg, log)

        elif cmd == "/help":
            log.write(
                "[bold cyan]Forge Commands:[/]\n"
                "  [bold]/auto[/]              — Toggle auto-approve mode\n"
                "  [bold]/model[/]             — List installed models (numbered)\n"
                "  [bold]/model <#>[/]         — Switch to model by number\n"
                "  [bold]/model <name>[/]      — Switch to model by name\n"
                "  [bold]/models[/]            — Browse model categories\n"
                "  [bold]/models <category>[/] — Show top models (coding, linux, general, writing)\n"
                "  [bold]/models pull <#|name>[/] — Download a model (number from category list)\n"
                "  [bold]/models update[/]     — Update all installed models\n"
                "  [bold]/project[/]           — Show project context and recent actions\n"
                "  [bold]/project init[/]      — Create a project for current directory\n"
                "  [bold]/project init <path>[/] — Create a project for a specific directory\n"
                "  [bold]/project clear[/]     — Reset project memory\n"
                "  [bold]/project rename[/]    — Set custom project name\n"
                "  [bold]/temp[/]              — Show/set temperature (0.0-2.0)\n"
                "  [bold]/ctx[/]               — Show/set context window size (tokens)\n"
                "  [bold]/maxtokens[/]         — Show/set max output tokens\n"
                "  [bold]/stats[/]             — Show detailed resource usage\n"
                "  [bold]/dashboard[/]         — Toggle dashboard panel (also F1)\n"
                "  [bold]/copy[/]              — Copy last AI response to clipboard\n"
                "  [bold]/copylog[/]           — Copy full session log to clipboard\n"
                "  [bold]/copylog <N>[/]       — Copy last N lines of log\n"
                "  [bold]/savelog[/]           — Save session log to ~/.config/forge/session.log\n"
                "  [bold]/clear[/]             — Clear chat history\n"
                "  [bold]/cwd <path>[/]        — Change working directory\n"
                "  [bold]/exit[/]              — Quit Forge (also /quit or Ctrl+Q)\n"
                "  [bold]/help[/]              — Show this help\n"
                "\n"
                "[bold cyan]Keyboard:[/]\n"
                "  [bold]Enter[/]              — Send message\n"
                "  [bold]Shift+Enter[/]        — New line in input\n"
                "  [bold]Ctrl+C[/]             — Cancel current operation\n"
                "  [bold]F1[/]                 — Toggle dashboard panel\n"
                "  [bold]Y/N/A/C[/]            — Approve/Reject/Auto-approve/Cancel in modal\n"
                "\n"
                "[bold cyan]Shell:[/]\n"
                "  [bold]!<command>[/]          — Run any shell command directly (e.g. !ls -la)\n"
                "  Common commands (cd, ls, git, pip, etc.) run directly without !\n"
                "\n"
                "[bold cyan]Tips:[/]\n"
                "  Click any [bold cyan]▶ command[/] in output to load it into input\n"
                "  Hold [bold]Shift[/] while selecting text for native terminal copy\n"
            )
        else:
            log.write(f"[bold red]Unknown command:[/] {cmd}. Type /help for available commands.")

    def _handle_project_command(self, arg: str, log: RichLog) -> None:
        """Handle /project — init, show, clear, or rename project memory."""
        proj = self.agent.project

        sub_parts = arg.split(maxsplit=1) if arg else []
        subcmd = sub_parts[0].lower() if sub_parts else ""

        if subcmd == "init":
            # /project init [path] — create a new project
            target = sub_parts[1].strip() if len(sub_parts) > 1 else ""
            if target:
                target = os.path.abspath(os.path.expanduser(target))
                if not os.path.isdir(target):
                    log.write(f"[bold red]Not a directory:[/] {target}")
                    return
            else:
                target = self.agent.shell.cwd

            from forge.project import ProjectMemory
            if ProjectMemory.exists(target):
                # Load it — this also re-homes copied projects automatically
                self.agent.load_project(target)
                loaded = self.agent.project
                if loaded:
                    log.write(f"[bold green]✓ Project loaded:[/] {loaded.name}")
                    log.write(f"[bold]Directory:[/] {loaded.directory}")
                    self._offer_gitignore(target, log)
                    self._refresh_project_button()
                    self._refresh_status()
                else:
                    log.write(f"[bold red]Failed to load project from:[/] {target}")
                return

            new_proj = self.agent.init_project(target)
            log.write(f"[bold green]✓ Project created:[/] {new_proj.name}")
            log.write(f"[bold]Directory:[/] {new_proj.directory}")
            log.write(f"[dim]Memory stored in:[/] {target}/.forge-memory/")
            self._offer_gitignore(target, log)
            self._refresh_project_button()
            self._refresh_status()
            return

        # All other subcommands require an active project
        if proj is None:
            log.write("[bold yellow]No active project.[/]")
            log.write(f"Create one with: [bold]/project init[/]  (for {self.agent.shell.cwd})")
            log.write("Or: [bold]/project init <path>[/]  for a different directory")
            return

        if not subcmd:
            # /project — show project context
            display = proj.format_display()
            for line in display.split("\n"):
                log.write(line)

        elif subcmd == "clear":
            proj.clear()
            proj.save()
            log.write("[bold green]✓ Project memory cleared.[/]")
            self._refresh_project_button()

        elif subcmd == "rename":
            new_name = sub_parts[1].strip() if len(sub_parts) > 1 else ""
            if not new_name:
                log.write("[bold red]Usage:[/] /project rename <name>")
                return
            proj.name = new_name
            proj.save()
            log.write(f"[bold green]✓ Project renamed to:[/] {new_name}")
            self._refresh_project_button()
            self._refresh_status()

        else:
            log.write(f"[bold red]Unknown subcommand:[/] {subcmd}")
            log.write("Usage: /project [init [path] | clear | rename <name>]")

    def _handle_model_command(self, arg: str, log: RichLog) -> None:
        """Handle /model — list installed models or switch by name/number."""
        from forge.llm import is_cloud_model, parse_provider
        from forge.models import CATEGORIES, get_installed_model_names

        installed = sorted(get_installed_model_names())

        if not arg:
            # Show numbered list of installed + available cloud models
            all_models: list[str] = list(installed)
            # Append cloud models that have a configured key
            for entry in CATEGORIES.get("online", {}).get("models", []):
                prov, _ = parse_provider(entry.name)
                if self._get_api_key_for_provider(prov):
                    if entry.name not in all_models:
                        all_models.append(entry.name)

            if not all_models:
                log.write("[bold red]No models available.[/] Use /models pull <name> to install one.")
                return
            self._model_menu = all_models
            log.write(f"[bold cyan]🤖 Available Models[/]  (active: [bold green]{self.config.model}[/])")
            for i, name in enumerate(all_models, 1):
                marker = " [bold green]◀ active[/]" if name == self.config.model else ""
                tag = " [bold blue]☁[/]" if is_cloud_model(name) else ""
                log.write(f"  [bold]{i}.[/] {name}{tag}{marker}")
            log.write("")
            log.write("[dim]Switch: /model <number>   (e.g. /model 1)[/]")
            log.write("[dim]Browse more: /models online   or   /models coding[/]")
            return

        # Check if arg is a number
        if arg.isdigit():
            idx = int(arg) - 1
            if not self._model_menu:
                # Rebuild menu on the fly
                all_models = list(installed)
                for entry in CATEGORIES.get("online", {}).get("models", []):
                    prov, _ = parse_provider(entry.name)
                    if self._get_api_key_for_provider(prov):
                        if entry.name not in all_models:
                            all_models.append(entry.name)
                self._model_menu = all_models
            if 0 <= idx < len(self._model_menu):
                arg = self._model_menu[idx]
            else:
                log.write(f"[bold red]Invalid number.[/] Run /model to see the list (1-{len(self._model_menu)}).")
                return

        # Auto-correct cloud model names missing the provider prefix
        if not is_cloud_model(arg):
            cloud_names: dict[str, str] = {}  # bare_name -> full prefixed name
            for entry in CATEGORIES.get("online", {}).get("models", []):
                _, bare = parse_provider(entry.name)
                cloud_names[bare] = entry.name
            bare = arg.split(":")[0]  # strip :latest tag if present
            if bare in cloud_names:
                arg = cloud_names[bare]
                log.write(f"[dim]→ Using cloud model: {arg}[/]")

        # Cloud models: check per-provider API key
        if is_cloud_model(arg):
            provider, _ = parse_provider(arg)
            if not self._get_api_key_for_provider(provider):
                self._prompt_api_key(provider, arg)
                return
            self.agent.change_model(arg)
            self.config.model = arg
            self._cached_models = None
            self.config.save()
            log.write(f"[bold green]✓ Switched to cloud model:[/] {arg} [bold blue]☁[/]")
            self._refresh_status()
            self._sync_select("#model-select", arg)
            return

        # Local model: validate that it's installed
        if arg not in installed:
            log.write(f"[bold red]Model not installed:[/] {arg}")
            clickable = self._make_clickable(f"/models pull {arg}")
            log.write(f"Install it? → {clickable}  (click or press Enter)")
            try:
                input_field = self.query_one("#input-field", ChatInput)
                input_field.load_text(f"/models pull {arg}")
                input_field.focus()
            except NoMatches:
                pass
            return

        # Switch to local model
        self.agent.change_model(arg)
        self.config.model = arg
        self._cached_models = None
        self.config.save()
        log.write(f"[bold green]✓ Switched to model:[/] {arg}")
        self._refresh_status()
        self._sync_select("#model-select", arg)

    def _handle_models_command(self, arg: str, log: RichLog) -> None:
        """Handle /models subcommands."""
        from forge.models import CATEGORIES, ModelEntry

        if not arg:
            output = format_all_categories(
                gpu_vram_gb=self._sys_info.gpu.vram_total_gb if self._sys_info.gpu else 0
            )
            for line in output.split("\n"):
                log.write(line)
            return

        sub_parts = arg.split(maxsplit=1)
        subcmd = sub_parts[0].lower()
        subarg = sub_parts[1].strip() if len(sub_parts) > 1 else ""

        if subcmd == "pull":
            if not subarg:
                log.write("[bold red]Usage:[/] /models pull <model_name or number>")
                return
            # Support pull by number from last category display
            if subarg.isdigit():
                idx = int(subarg) - 1
                if 0 <= idx < len(self._category_menu):
                    subarg = self._category_menu[idx]
                else:
                    log.write(f"[bold red]Invalid number.[/] Browse a category first: /models coding")
                    return
            log.write(f"[bold yellow]Pulling model:[/] {subarg}...")
            self._pull_model_async(subarg)

        elif subcmd == "update":
            models = update_all_models()
            if not models:
                log.write("[bold red]No installed models found.[/]")
                return
            log.write(f"[bold yellow]Updating {len(models)} models:[/] {', '.join(models)}")
            for model_name in models:
                self._pull_model_async(model_name)

        elif subcmd in CATEGORIES:
            # Show category with numbered list, store for pull-by-number
            vram = self._sys_info.gpu.vram_total_gb if self._sys_info.gpu else 0
            cat_models = CATEGORIES[subcmd]["models"]
            self._category_menu = [m.name for m in cat_models]
            output = format_category(subcmd, gpu_vram_gb=vram)
            for line in output.split("\n"):
                log.write(line)
            if subcmd == "online":
                # Cloud models don't need pulling — set _model_menu so /model <N> works
                self._model_menu = self._category_menu
                log.write("[dim]Switch: /model <number>   (e.g. /model 2)[/]")
            else:
                log.write("[dim]Install by number: /models pull <number>   (e.g. /models pull 2)[/]")

        else:
            available = ", ".join(CATEGORIES.keys())
            log.write(f"[bold red]Unknown subcommand or category:[/] {subcmd}")
            log.write(f"Categories: {available}")
            log.write("Also: /models pull <name>, /models update")

    @work(thread=True)
    def _pull_model_async(self, name: str) -> None:
        """Pull a model in the background and stream progress."""
        try:
            proc = pull_model(name)
            for line in proc.stdout:
                line = line.strip()
                if line:
                    self.post_message(AgentEventMessage(
                        AgentEvent(kind="tool_result", content=f"[dim]{line}[/]")
                    ))
            proc.wait()
            if proc.returncode == 0:
                self._cached_models = None  # invalidate so dashboard refreshes
                # Rebuild the model menu and refresh dropdown
                self.call_from_thread(self._refresh_model_dropdown)
                self.post_message(AgentEventMessage(
                    AgentEvent(kind="text",
                               content=f"\n[bold green]✓ Model {name} is ready![/]")
                ))
                self.post_message(AgentEventMessage(
                    AgentEvent(kind="text",
                               content=f"Switch to it? → /model {name}  (press Enter)\n")
                ))
                # Pre-fill input so user just presses Enter to switch
                self.call_from_thread(self._prefill_input, f"/model {name}")
            else:
                self.post_message(AgentEventMessage(
                    AgentEvent(kind="error", content=f"Failed to pull {name}")
                ))
        except Exception as e:
            self.post_message(AgentEventMessage(
                AgentEvent(kind="error", content=f"Error pulling {name}: {e}")
            ))

    def _prefill_input(self, text: str) -> None:
        """Pre-fill the input field (must be called on the main thread)."""
        try:
            input_field = self.query_one("#input-field", ChatInput)
            input_field.load_text(text)
            input_field.focus()
        except NoMatches:
            pass

    def _refresh_status(self) -> None:
        try:
            self.query_one("#status-bar", Static).update(self._build_status())
        except NoMatches:
            pass

    def _sync_select(self, selector: str, value) -> None:
        """Update a Select widget's value without re-triggering the change handler."""
        try:
            self._updating_widgets = True
            sel = self.query_one(selector, Select)
            if sel.value != value:
                sel.value = value
        except (NoMatches, Exception):
            pass
        finally:
            self._updating_widgets = False

    # ─── Agent Worker ─────────────────────────────────────────────────────

    @work(thread=True)
    def _run_agent(self, user_input: str) -> None:
        """Run the agent in a background thread."""
        try:
            self.agent.process_message(
                user_input=user_input,
                on_event=self._on_agent_event,
                get_approval=self._get_approval_blocking,
            )
        except Exception as e:
            self.post_message(AgentEventMessage(
                AgentEvent(kind="error", content=f"Agent error: {e}")
            ))
        finally:
            self.post_message(AgentDoneMessage())

    def _on_agent_event(self, event: AgentEvent) -> None:
        self.post_message(AgentEventMessage(event))

    def _get_approval_blocking(
        self, tool_call: ToolCall, safety: SafetyResult
    ) -> ApprovalAction:
        result_event = threading.Event()
        msg = ApprovalRequestMessage(tool_call, safety, result_event)
        self.post_message(msg)
        result_event.wait()
        return msg.action

    # ─── Message Handlers (main thread) ───────────────────────────────────

    @on(AgentEventMessage)
    def on_agent_event(self, message: AgentEventMessage) -> None:
        """Handle agent events on the main thread."""
        event = message.event
        log = self.query_one("#chat-log", RichLog)

        if event.kind == "thinking":
            # Stream thinking tokens in dim italic
            if not hasattr(self, "_thinking_buffer"):
                self._thinking_buffer = ""
                self._thinking_started = False

            if not self._thinking_started:
                self._thinking_buffer = "[dim italic #888888]🧠 "
                self._thinking_started = True

            self._thinking_buffer += rich_escape(event.content)
            while "\n" in self._thinking_buffer:
                line, self._thinking_buffer = self._thinking_buffer.split("\n", 1)
                log.write(f"{line}[/]")
                # Re-open style tag on next line
                self._thinking_buffer = "[dim italic #888888]" + self._thinking_buffer

        elif event.kind == "thinking_done":
            # Flush remaining thinking text and close the section
            if hasattr(self, "_thinking_buffer") and self._thinking_buffer:
                log.write(f"{self._thinking_buffer}[/]")
                self._thinking_buffer = ""
            self._thinking_started = False
            log.write("[dim]───[/]")

        elif event.kind == "text":
            if not hasattr(self, "_stream_buffer"):
                self._stream_buffer = ""
                self._stream_started = False

            if not self._stream_started:
                self._stream_buffer = "[bold #00ff88]Forge:[/] "
                self._stream_started = True

            self._last_response_text += event.content
            self._stream_buffer += rich_escape(event.content)
            while "\n" in self._stream_buffer:
                line, self._stream_buffer = self._stream_buffer.split("\n", 1)
                log.write(self._linkify_commands(line))

        elif event.kind == "tool_call":
            self._flush_stream(log)
            cmd = event.content
            clickable = self._make_clickable(cmd)
            if event.safety and event.safety.level == RiskLevel.DANGEROUS:
                log.write(f"\n[bold #ff8800]⚠ Command:[/] [on #332200]{clickable}[/]")
            else:
                log.write(f"\n[bold #ffaa00]⚡ Running:[/] {clickable}")

        elif event.kind == "tool_output":
            # Real-time streaming line from a running command
            log.write(Text(event.content, style="#888888"))

        elif event.kind == "tool_result":
            self._flush_stream(log)
            content = event.content
            if len(content) > 2000:
                content = content[:2000] + "\n... [truncated for display]"
            log.write(f"[#888888]{rich_escape(content)}[/]")

        elif event.kind == "error":
            self._flush_stream(log)
            log.write(f"\n[bold #ff4444]Error:[/] {event.content}")

    def _flush_stream(self, log: RichLog) -> None:
        if hasattr(self, "_stream_buffer") and self._stream_buffer:
            log.write(self._linkify_commands(self._stream_buffer))
            self._stream_buffer = ""
        self._stream_started = False

    @on(AgentDoneMessage)
    def on_agent_done(self, message: AgentDoneMessage) -> None:
        log = self.query_one("#chat-log", RichLog)
        self._flush_stream(log)
        self.is_processing = False
        # Hide the active prompt display
        try:
            prompt_display = self.query_one("#active-prompt", Static)
            prompt_display.update("")
            prompt_display.styles.display = "none"
        except NoMatches:
            pass
        # Auto-save project memory after each agent run
        try:
            self.agent.save_project()
        except Exception:
            pass
        self.query_one("#input-field", ChatInput).focus()

    @on(ApprovalRequestMessage)
    def on_approval_request(self, message: ApprovalRequestMessage) -> None:
        """Show approval modal and return result to agent thread."""
        self._pending_approval = message

        tc = message.tool_call
        if tc.name == "run_command":
            cmd_text = tc.arguments.get("command", str(tc.arguments))
        else:
            import json
            cmd_text = f"{tc.name}: {json.dumps(tc.arguments, indent=2)}"

        def on_dismiss(action: ApprovalAction) -> None:
            if self._pending_approval:
                self._pending_approval.action = action
                self._pending_approval.result_event.set()
                self._pending_approval = None
                # Sync auto-approve to the app reactive so it persists
                # and no more popups appear for safe commands
                if action == ApprovalAction.AUTO_APPROVE:
                    self.auto_approve = True
                elif action == ApprovalAction.CANCEL:
                    self.agent.cancel()

        is_sudo = (
            tc.name == "run_command"
            and tc.arguments.get("command", "").strip().startswith("sudo ")
        )

        self.push_screen(
            ApprovalModal(
                command=cmd_text,
                safety=message.safety,
                is_command=(tc.name == "run_command"),
                is_sudo=is_sudo,
            ),
            callback=on_dismiss,
        )

    # ─── Actions ──────────────────────────────────────────────────────────

    def action_click_command(self, cmd_id: str) -> None:
        """Handle click on a suggested command — load it into the input field."""
        cmd = self._clickable_commands.get(int(cmd_id))
        if cmd:
            try:
                input_field = self.query_one("#input-field", ChatInput)
                input_field.load_text(cmd)
                input_field.focus()
            except NoMatches:
                pass

    def action_copy_last(self) -> None:
        """Copy last agent response to clipboard."""
        if self._last_response_text.strip():
            if self._copy_to_clipboard(self._last_response_text.strip()):
                self.notify("Copied to clipboard!", severity="information", timeout=2)
            else:
                self.notify("Clipboard not available", severity="warning", timeout=2)
        else:
            self.notify("Nothing to copy", severity="warning", timeout=2)

    def action_toggle_dashboard(self) -> None:
        """Toggle the dashboard panel visibility."""
        try:
            dashboard = self.query_one("#dashboard", Vertical)
            if dashboard.styles.display == "none":
                dashboard.styles.display = "block"
                self._update_resources()  # refresh immediately on show
            else:
                dashboard.styles.display = "none"
        except NoMatches:
            pass

    def action_quit_app(self) -> None:
        """Quit the application."""
        self.exit()

    def action_cancel(self) -> None:
        if self.is_processing:
            self.agent.cancel()
            log = self.query_one("#chat-log", RichLog)
            self._flush_stream(log)
            log.write("\n[bold red]Cancelled.[/]")
            # Restore the last command into the input so user can edit and retry
            if self._last_input:
                input_field = self.query_one("#input-field", ChatInput)
                input_field.restore(self._last_input)


def main() -> None:
    """Entry point for the Forge application."""
    parser = argparse.ArgumentParser(description="Forge — Local AI Coding Terminal")
    parser.add_argument("--model", "-m", help="Ollama model to use")
    parser.add_argument("--cwd", "-d", help="Working directory")
    parser.add_argument("--host", help="Ollama server host (default: http://localhost:11434)")
    args = parser.parse_args()

    config = ForgeConfig.load(
        cli_model=args.model,
        cli_cwd=args.cwd,
    )
    if args.host:
        config.ollama_host = args.host

    app = ForgeApp(config)
    app.run()


if __name__ == "__main__":
    main()

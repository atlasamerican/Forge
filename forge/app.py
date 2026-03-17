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
    Label,
    RichLog,
    Static,
    TextArea,
)

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
        max-height: 20;
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
    ) -> None:
        super().__init__()
        self.command_text = command
        self.safety = safety
        self.is_command = is_command

    def compose(self) -> ComposeResult:
        title = "🔒 Command Approval Required" if self.is_command else "📋 Operation Approval"
        risk_text = f"⚠ {self.safety.reason}" if self.safety.level == RiskLevel.DANGEROUS else ""

        with Vertical(id="approval-dialog"):
            yield Label(title, id="approval-title")
            yield Static(self.command_text, id="approval-command")
            if risk_text:
                yield Label(risk_text, id="approval-risk")
            with Horizontal(id="approval-buttons"):
                yield Button("[Y] Approve", variant="success", id="btn-approve")
                yield Button("[N] Reject", variant="error", id="btn-reject")
                yield Button("[A] Auto-approve", variant="warning", id="btn-auto")

    def action_approve(self) -> None:
        self.dismiss(ApprovalAction.APPROVE)

    def action_reject(self) -> None:
        self.dismiss(ApprovalAction.REJECT)

    def action_auto_approve(self) -> None:
        self.dismiss(ApprovalAction.AUTO_APPROVE)

    @on(Button.Pressed, "#btn-approve")
    def on_approve(self) -> None:
        self.dismiss(ApprovalAction.APPROVE)

    @on(Button.Pressed, "#btn-reject")
    def on_reject(self) -> None:
        self.dismiss(ApprovalAction.REJECT)

    @on(Button.Pressed, "#btn-auto")
    def on_auto(self) -> None:
        self.dismiss(ApprovalAction.AUTO_APPROVE)


# ─── Custom Messages ─────────────────────────────────────────────────────────


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

    #dashboard-content {
        height: auto;
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

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="dashboard"):
            yield Static("", id="dashboard-content")
        yield RichLog(id="chat-log", wrap=True, highlight=True, markup=True)
        with Horizontal(id="input-bar"):
            yield ChatInput(id="input-field")
        yield Static(self._build_status(), id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app with system info and start resource monitor."""
        self.auto_approve = self.config.auto_approve
        log = self.query_one("#chat-log", RichLog)

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
        return f" {mode} | 🤖 {model} | 📁 {cwd} | {status}"

    def _update_resources(self) -> None:
        """Periodic update of the dashboard panel."""
        try:
            dashboard = self.query_one("#dashboard", Vertical)
            if dashboard.styles.display == "none":
                return  # skip work when hidden
            stats = self._monitor.snapshot()
            content = self.query_one("#dashboard-content", Static)
            content.update(self._build_dashboard(stats))
        except Exception:
            pass

    def _build_dashboard(self, stats=None) -> str:
        """Build the dashboard panel content with Rich markup."""
        if stats is None:
            stats = self._monitor.snapshot()

        lines = []
        # Resource bars
        lines.append(f"  [bold]CPU[/]  {stats.cpu_percent:5.1f}%  {_bar(stats.cpu_percent, 12)}"
                     f"      [bold]RAM[/]  {stats.ram_used_gb:.0f}/{stats.ram_total_gb:.0f} GB"
                     f" ({stats.ram_percent:.0f}%)  {_bar(stats.ram_percent, 12)}")
        gpu_line = ""
        if stats.gpu_busy_percent is not None:
            gpu_line = f"  [bold]GPU[/]  {stats.gpu_busy_percent:5.1f}%  {_bar(stats.gpu_busy_percent, 12)}"
        if stats.vram_used_gb is not None and stats.vram_total_gb is not None:
            gpu_line += f"      [bold]VRAM[/] {stats.vram_used_gb:.1f}/{stats.vram_total_gb:.1f} GB"
            gpu_line += f" ({stats.vram_percent:.0f}%)  {_bar(stats.vram_percent, 12)}"
        if gpu_line:
            lines.append(gpu_line)

        lines.append("")

        # Model info
        lines.append(f"  [bold]🤖 Active:[/] [bold cyan]{self.config.model}[/]")
        # Refresh model list every ~30s (15 ticks × 2s interval) instead of every tick
        self._models_cache_tick += 1
        if self._cached_models is None or self._models_cache_tick >= 15:
            self._cached_models = get_ollama_models()
            self._models_cache_tick = 0
        if self._cached_models:
            names = [m['name'] for m in self._cached_models]
            lines.append(f"  [bold]📦 Installed:[/] {', '.join(names)}")

        lines.append("")

        # Quick commands reference
        mode = "[bold green]AUTO[/]" if self.auto_approve else "[bold red]MANUAL[/]"
        lines.append(f"  Mode: {mode}"
                     f"  |  [bold]/model[/] <name> switch"
                     f"  |  [bold]/models[/] coding browse"
                     f"  |  [bold]/models pull[/] <name> install"
                     f"  |  [bold]F1[/] hide")

        return "\n".join(lines)

    def watch_auto_approve(self, value: bool) -> None:
        self.config.auto_approve = value
        self.agent.config.auto_approve = value
        try:
            self.query_one("#status-bar", Static).update(self._build_status())
        except NoMatches:
            pass

    def watch_is_processing(self, value: bool) -> None:
        try:
            self.query_one("#status-bar", Static).update(self._build_status())
            self.query_one("#input-field", ChatInput).disabled = value
        except NoMatches:
            pass

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

    @on(ChatInput.Submitted)
    def on_chat_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle user input from the chat input widget."""
        text = event.text.strip()
        if not text:
            return

        if text.startswith("/"):
            self._handle_slash_command(text)
            return

        # Remember this input so we can restore it on cancel
        self._last_input = text
        self._last_response_text = ""  # reset for new response

        log = self.query_one("#chat-log", RichLog)
        log.write(f"\n[bold #00aaff]You:[/] {text}")

        self.is_processing = True
        self._run_agent(text)

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
                    self.agent.shell.cwd = path
                    self.agent.update_system_prompt()
                    log.write(f"[bold green]Working directory:[/] {path}")
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

        elif cmd == "/stats":
            stats = self._monitor.snapshot()
            detailed = self._monitor.format_detailed(stats)
            for line in detailed.split("\n"):
                log.write(line)

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
                "  [bold]/stats[/]             — Show detailed resource usage\n"
                "  [bold]/dashboard[/]         — Toggle dashboard panel (also F1)\n"
                "  [bold]/copy[/]              — Copy last AI response to clipboard\n"
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
                "  [bold]Y/N/A[/]              — Approve/Reject/Auto-approve in modal\n"
                "\n"
                "[bold cyan]Tips:[/]\n"
                "  Click any [bold cyan]▶ command[/] in output to load it into input\n"
                "  Hold [bold]Shift[/] while selecting text for native terminal copy\n"
            )
        else:
            log.write(f"[bold red]Unknown command:[/] {cmd}. Type /help for available commands.")

    def _handle_model_command(self, arg: str, log: RichLog) -> None:
        """Handle /model — list installed models or switch by name/number."""
        from forge.llm import is_cloud_model
        from forge.models import get_installed_model_names

        installed = sorted(get_installed_model_names())

        if not arg:
            # Show numbered list of installed + available cloud models
            all_models: list[str] = list(installed)
            # Append cloud models that are always available
            cloud_models = [
                "gemini:gemini-2.0-flash",
                "gemini:gemini-2.5-pro-preview-03-25",
            ]
            if self.config.gemini_api_key:
                for cm in cloud_models:
                    if cm not in all_models:
                        all_models.append(cm)

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
                if self.config.gemini_api_key:
                    all_models.extend([
                        "gemini:gemini-2.0-flash",
                        "gemini:gemini-2.5-pro-preview-03-25",
                    ])
                self._model_menu = all_models
            if 0 <= idx < len(self._model_menu):
                arg = self._model_menu[idx]
            else:
                log.write(f"[bold red]Invalid number.[/] Run /model to see the list (1-{len(self._model_menu)}).")
                return

        # Cloud models: just need API key, no install check
        if is_cloud_model(arg):
            if not self.config.gemini_api_key:
                log.write(f"[bold red]API key required for {arg}[/]")
                log.write("Set it in [bold]~/.config/forge/config.toml[/]:")
                log.write('  [api_keys]')
                log.write('  gemini = "your-api-key-here"')
                return
            self.agent.change_model(arg)
            self.config.model = arg
            self._cached_models = None
            log.write(f"[bold green]✓ Switched to cloud model:[/] {arg} [bold blue]☁[/]")
            self._refresh_status()
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
        log.write(f"[bold green]✓ Switched to model:[/] {arg}")
        self._refresh_status()

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
                # Rebuild the model menu so /model <N> works with fresh list
                from forge.models import get_installed_model_names
                self._model_menu = sorted(get_installed_model_names())
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

        if event.kind == "text":
            if not hasattr(self, "_stream_buffer"):
                self._stream_buffer = ""
                self._stream_started = False

            if not self._stream_started:
                self._stream_buffer = "[bold #00ff88]Forge:[/] "
                self._stream_started = True

            self._last_response_text += event.content
            self._stream_buffer += event.content
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

        elif event.kind == "tool_result":
            self._flush_stream(log)
            content = event.content
            if len(content) > 2000:
                content = content[:2000] + "\n... [truncated for display]"
            log.write(f"[#888888]{content}[/]")

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

        self.push_screen(
            ApprovalModal(
                command=cmd_text,
                safety=message.safety,
                is_command=(tc.name == "run_command"),
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

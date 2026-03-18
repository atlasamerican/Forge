"""Environment Instructions & Reference Documents for Forge.

Two-tier system:
  - Global instructions:  ~/.config/forge/instructions/
  - Project instructions: <project>/.forge-memory/instructions/

Instructions are behavioral guidelines injected into the LLM system prompt.
Reference documents are files the LLM can read on demand (HTML templates, docs, etc.).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ─── Paths ────────────────────────────────────────────────────────────────────

GLOBAL_INSTRUCTIONS_DIR = Path.home() / ".config" / "forge" / "instructions"
GLOBAL_REFERENCES_DIR = Path.home() / ".config" / "forge" / "references"

# Project-level dirs are relative to .forge-memory/
PROJECT_INSTRUCTIONS_SUBDIR = "instructions"
PROJECT_REFERENCES_SUBDIR = "references"

# Max bytes to read for a reference document summary line
_REF_PREVIEW_BYTES = 200


# ─── Built-in instruction catalog ─────────────────────────────────────────────

BUILTIN_INSTRUCTIONS: list[dict[str, Any]] = [
    {
        "id": "no-hardcoded-secrets",
        "title": "No Hardcoded Secrets",
        "text": (
            "Never hardcode passwords, API keys, tokens, or credentials in source code. "
            "Use environment variables, config files (added to .gitignore), or secret managers instead. "
            "If you find existing hardcoded secrets during review, flag them immediately and suggest safer alternatives."
        ),
        "enabled": True,
        "builtin": True,
    },
    {
        "id": "git-branch-workflow",
        "title": "Git Branch Workflow",
        "text": (
            "Work on the development or feature branch, never directly on main/master. "
            "After changes are tested and confirmed by the user, prompt to merge or sync to the main/stable branch. "
            "Never push directly to main without explicit user approval."
        ),
        "enabled": True,
        "builtin": True,
    },
    {
        "id": "code-documentation",
        "title": "Code Documentation",
        "text": (
            "Include docstrings and comments for all public functions, classes, and modules. "
            "Update the README when adding significant features. "
            "Include usage examples where helpful."
        ),
        "enabled": True,
        "builtin": True,
    },
    {
        "id": "error-handling",
        "title": "Error Handling",
        "text": (
            "Always include proper error handling and input validation. "
            "Never silently swallow exceptions. "
            "Provide meaningful error messages that help with debugging."
        ),
        "enabled": True,
        "builtin": True,
    },
    {
        "id": "security-first",
        "title": "Security First",
        "text": (
            "Validate all user inputs. Use parameterized queries for databases. "
            "Avoid shell injection risks by using argument lists instead of string concatenation. "
            "Follow the principle of least privilege."
        ),
        "enabled": True,
        "builtin": True,
    },
    {
        "id": "test-coverage",
        "title": "Test Coverage",
        "text": (
            "Write tests for new functionality. Run existing tests before and after changes. "
            "Don't break existing test suites. Suggest test strategies when building new features."
        ),
        "enabled": False,
        "builtin": True,
    },
    {
        "id": "clean-code",
        "title": "Clean Code",
        "text": (
            "Follow the project's existing code style and conventions. "
            "Keep functions small and focused. Avoid code duplication. "
            "Use meaningful variable and function names."
        ),
        "enabled": True,
        "builtin": True,
    },
    {
        "id": "backup-before-modify",
        "title": "Backup Before Modify",
        "text": (
            "Before modifying existing files, read and understand them first. "
            "When making significant changes to critical files, suggest creating a backup or working on a copy. "
            "Prefer incremental changes over large rewrites."
        ),
        "enabled": False,
        "builtin": True,
    },
]


# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class Instruction:
    """A single environment instruction."""

    id: str
    title: str
    text: str
    enabled: bool = True
    builtin: bool = False
    scope: str = "global"  # "global" or "project"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "enabled": self.enabled,
            "builtin": self.builtin,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], scope: str = "global") -> Instruction:
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            text=data.get("text", ""),
            enabled=data.get("enabled", True),
            builtin=data.get("builtin", False),
            scope=scope,
        )


@dataclass
class ReferenceDoc:
    """A reference document available for the LLM to read."""

    path: str  # absolute path
    name: str  # filename
    preview: str  # first few lines
    scope: str = "global"  # "global" or "project"


# ─── Instruction Manager ─────────────────────────────────────────────────────


class InstructionManager:
    """Manages environment instructions and reference documents.

    Loads from global dir (~/.config/forge/) and optionally from a
    project's .forge-memory/ directory. Global and project instructions
    are merged, with project-level overrides taking precedence for
    matching IDs.
    """

    def __init__(self, project_dir: str | None = None) -> None:
        self._project_dir = project_dir
        self._instructions: list[Instruction] = []
        self._references: list[ReferenceDoc] = []
        self._load()

    @property
    def instructions(self) -> list[Instruction]:
        return self._instructions

    @property
    def references(self) -> list[ReferenceDoc]:
        return self._references

    def set_project_dir(self, project_dir: str | None) -> None:
        """Change the project directory and reload everything."""
        self._project_dir = project_dir
        self._load()

    # ─── Loading ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load all instructions and references from disk."""
        self._instructions = []
        self._references = []

        # 1. Seed with built-in instructions
        builtins = {b["id"]: Instruction.from_dict(b, scope="global") for b in BUILTIN_INSTRUCTIONS}

        # 2. Load global overrides (user may have toggled enabled state)
        global_dir = GLOBAL_INSTRUCTIONS_DIR
        global_overrides = self._load_instructions_dir(global_dir, scope="global")

        # Merge: global overrides win over builtins
        merged: dict[str, Instruction] = dict(builtins)
        for inst in global_overrides:
            merged[inst.id] = inst

        # 3. Load project-level instructions
        project_instructions: list[Instruction] = []
        if self._project_dir:
            proj_inst_dir = Path(self._project_dir) / ".forge-memory" / PROJECT_INSTRUCTIONS_SUBDIR
            project_instructions = self._load_instructions_dir(proj_inst_dir, scope="project")
            for inst in project_instructions:
                merged[inst.id] = inst

        # Build final sorted list: builtins first (in catalog order), then custom
        builtin_ids = [b["id"] for b in BUILTIN_INSTRUCTIONS]
        result: list[Instruction] = []
        for bid in builtin_ids:
            if bid in merged:
                result.append(merged.pop(bid))
        # Remaining are custom instructions, sorted by title
        result.extend(sorted(merged.values(), key=lambda i: i.title))
        self._instructions = result

        # 4. Load reference documents
        self._references = []
        self._references.extend(self._scan_references(GLOBAL_REFERENCES_DIR, "global"))
        if self._project_dir:
            proj_ref_dir = Path(self._project_dir) / ".forge-memory" / PROJECT_REFERENCES_SUBDIR
            self._references.extend(self._scan_references(proj_ref_dir, "project"))

    def _load_instructions_dir(self, directory: Path, scope: str) -> list[Instruction]:
        """Load instruction JSON files from a directory."""
        instructions: list[Instruction] = []
        if not directory.is_dir():
            return instructions
        for path in sorted(directory.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                instructions.append(Instruction.from_dict(data, scope=scope))
            except Exception:
                continue
        return instructions

    def _scan_references(self, directory: Path, scope: str) -> list[ReferenceDoc]:
        """Scan a directory for reference documents."""
        docs: list[ReferenceDoc] = []
        if not directory.is_dir():
            return docs
        for path in sorted(directory.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                preview = self._read_preview(path)
                docs.append(ReferenceDoc(
                    path=str(path),
                    name=path.name,
                    preview=preview,
                    scope=scope,
                ))
        return docs

    @staticmethod
    def _read_preview(path: Path) -> str:
        """Read the first few lines of a file for preview."""
        try:
            with open(path, "r", errors="replace") as f:
                text = f.read(_REF_PREVIEW_BYTES)
            # Take first 3 lines max
            lines = text.splitlines()[:3]
            return " ".join(line.strip() for line in lines if line.strip())[:150]
        except Exception:
            return "(binary or unreadable)"

    # ─── Mutations ────────────────────────────────────────────────────────

    def toggle(self, instruction_id: str) -> bool | None:
        """Toggle an instruction's enabled state. Returns new state or None if not found."""
        for inst in self._instructions:
            if inst.id == instruction_id:
                inst.enabled = not inst.enabled
                self._save_instruction(inst)
                return inst.enabled
        return None

    def set_enabled(self, instruction_id: str, enabled: bool) -> None:
        """Set an instruction's enabled state."""
        for inst in self._instructions:
            if inst.id == instruction_id:
                inst.enabled = enabled
                self._save_instruction(inst)
                return

    def add(
        self,
        title: str,
        text: str,
        scope: str = "global",
        enabled: bool = True,
    ) -> Instruction:
        """Add a new custom instruction."""
        # Generate an ID from title
        inst_id = title.lower().replace(" ", "-")
        inst_id = "".join(c for c in inst_id if c.isalnum() or c == "-")[:50]
        # Ensure uniqueness
        existing_ids = {i.id for i in self._instructions}
        base_id = inst_id
        counter = 2
        while inst_id in existing_ids:
            inst_id = f"{base_id}-{counter}"
            counter += 1

        inst = Instruction(
            id=inst_id,
            title=title,
            text=text,
            enabled=enabled,
            builtin=False,
            scope=scope,
        )
        self._instructions.append(inst)
        self._save_instruction(inst)
        return inst

    def remove(self, instruction_id: str) -> bool:
        """Remove a custom instruction. Returns True if removed, False if not found or builtin."""
        for i, inst in enumerate(self._instructions):
            if inst.id == instruction_id:
                if inst.builtin:
                    return False  # can't delete builtins, only disable
                self._instructions.pop(i)
                self._delete_instruction_file(inst)
                return True
        return False

    def add_reference(self, source_path: str, scope: str = "global") -> ReferenceDoc | None:
        """Copy a file into the references directory. Returns the ReferenceDoc or None on failure."""
        src = Path(source_path)
        if not src.is_file():
            return None
        if scope == "project" and self._project_dir:
            dest_dir = Path(self._project_dir) / ".forge-memory" / PROJECT_REFERENCES_SUBDIR
        else:
            dest_dir = GLOBAL_REFERENCES_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        try:
            shutil.copy2(str(src), str(dest))
        except Exception:
            return None
        doc = ReferenceDoc(
            path=str(dest),
            name=dest.name,
            preview=self._read_preview(dest),
            scope=scope,
        )
        self._references.append(doc)
        return doc

    def remove_reference(self, name: str) -> bool:
        """Remove a reference document by filename."""
        for i, doc in enumerate(self._references):
            if doc.name == name:
                try:
                    Path(doc.path).unlink(missing_ok=True)
                except Exception:
                    pass
                self._references.pop(i)
                return True
        return False

    # ─── Persistence ──────────────────────────────────────────────────────

    def _save_instruction(self, inst: Instruction) -> None:
        """Save an instruction to its appropriate directory."""
        if inst.scope == "project" and self._project_dir:
            directory = Path(self._project_dir) / ".forge-memory" / PROJECT_INSTRUCTIONS_SUBDIR
        else:
            directory = GLOBAL_INSTRUCTIONS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{inst.id}.json"
        with open(path, "w") as f:
            json.dump(inst.to_dict(), f, indent=2)

    def _delete_instruction_file(self, inst: Instruction) -> None:
        """Delete the JSON file for an instruction."""
        if inst.scope == "project" and self._project_dir:
            directory = Path(self._project_dir) / ".forge-memory" / PROJECT_INSTRUCTIONS_SUBDIR
        else:
            directory = GLOBAL_INSTRUCTIONS_DIR
        path = directory / f"{inst.id}.json"
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    # ─── Formatting for system prompt ─────────────────────────────────────

    def format_instructions_for_prompt(self) -> str:
        """Format enabled instructions for injection into the system prompt."""
        enabled = [i for i in self._instructions if i.enabled]
        if not enabled:
            return ""
        lines = ["## Environment Instructions"]
        lines.append("Follow these guidelines in all your work:\n")
        for inst in enabled:
            lines.append(f"- **{inst.title}**: {inst.text}")
        return "\n".join(lines)

    def format_references_for_prompt(self) -> str:
        """Format reference documents listing for the system prompt."""
        if not self._references:
            return ""
        lines = ["## Reference Documents"]
        lines.append(
            "The following reference files are available. "
            "Use the read_file tool to access their full content when needed.\n"
        )
        for doc in self._references:
            scope_tag = " (project)" if doc.scope == "project" else ""
            lines.append(f"- `{doc.path}`{scope_tag} — {doc.preview}")
        return "\n".join(lines)

    # ─── Formatting for display ───────────────────────────────────────────

    def format_display(self) -> str:
        """Format instructions list for Rich markup display."""
        lines = ["[bold cyan]📋 Environment Instructions[/]", ""]
        for i, inst in enumerate(self._instructions, 1):
            check = "[bold green]✓[/]" if inst.enabled else "[dim]✗[/]"
            scope = " [dim](project)[/]" if inst.scope == "project" else ""
            builtin = " [dim](built-in)[/]" if inst.builtin else ""
            lines.append(f"  {check} [bold]{i}.[/] {inst.title}{scope}{builtin}")
            lines.append(f"      [dim]{inst.text[:80]}{'...' if len(inst.text) > 80 else ''}[/]")
        lines.append("")
        lines.append("[dim]Toggle: /instructions toggle <#>  |  Add: /instructions add <title>[/]")
        lines.append("[dim]Remove: /instructions remove <#>  |  Refs: /instructions refs[/]")
        return "\n".join(lines)

    def format_references_display(self) -> str:
        """Format reference documents for Rich markup display."""
        if not self._references:
            return (
                "[bold cyan]📁 Reference Documents[/]\n\n"
                "[dim]No reference documents found.[/]\n"
                f"[dim]Place files in:[/]\n"
                f"  [bold]Global:[/]  {GLOBAL_REFERENCES_DIR}/\n"
                f"  [bold]Project:[/] <project>/.forge-memory/references/\n\n"
                "[dim]Or use: /instructions refs add <filepath>[/]"
            )
        lines = ["[bold cyan]📁 Reference Documents[/]", ""]
        for i, doc in enumerate(self._references, 1):
            scope = "[dim](project)[/]" if doc.scope == "project" else "[dim](global)[/]"
            lines.append(f"  [bold]{i}.[/] {doc.name} {scope}")
            lines.append(f"      [dim]{doc.path}[/]")
            lines.append(f"      [dim]{doc.preview}[/]")
        lines.append("")
        lines.append("[dim]Add: /instructions refs add <filepath>[/]")
        lines.append("[dim]Remove: /instructions refs remove <name>[/]")
        return "\n".join(lines)

    def get_by_index(self, index: int) -> Instruction | None:
        """Get instruction by 1-based index."""
        if 1 <= index <= len(self._instructions):
            return self._instructions[index - 1]
        return None

    def get_by_id_or_title(self, query: str) -> Instruction | None:
        """Find instruction by ID or title (case-insensitive)."""
        q = query.lower()
        for inst in self._instructions:
            if inst.id == q or inst.title.lower() == q:
                return inst
        return None

    def enabled_count(self) -> int:
        """Return the number of enabled instructions."""
        return sum(1 for i in self._instructions if i.enabled)

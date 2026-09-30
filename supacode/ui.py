"""Terminal presentation layer.

Knows nothing about LLMs, providers or tools - it only receives plain
strings and dicts and decides how they look. Ported from neuralcode's
ui.py, trimmed to what supacode actually has right now: no session
persistence (no resumed/replay), no permission gate (no approve), no
subagents, no write_todos tool, no compaction, no context-reminder
injection. Those methods land here once their features exist - not
before, same rule as ToolSpec.kind.
"""

import json
from contextlib import contextmanager

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

ACCENT = "#7aa2f7"
USER = "#9ece6a"
TOOL = "#e0af68"
MUTED = "#565f89"

MAX_TOOL_OUTPUT_LINES = 12


class UI:
    def __init__(self):
        self.console = Console()
        self._totals = {}

    # ---------------------------------------------------------------- input

    def banner(self):
        self.console.print()
        self.console.print(
            Rule(Text(" coding agent ", style=f"bold {ACCENT}"), style=MUTED)
        )
        self.console.print(
            Padding(Text("ctrl-d or empty line to exit", style=MUTED), (0, 0, 0, 2))
        )

    def ask(self):
        self.console.print()
        try:
            return input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return ""

    # --------------------------------------------------------------- output

    def agent(self, text):
        self.console.print(
            Padding(
                Group(
                    Text("agent", style=f"bold {ACCENT}"),
                    Padding(Markdown(text.strip()), (1, 0, 0, 0)),
                ),
                (1, 2, 0, 2),
            )
        )

    def tool(self, name, args, result):
        header = Text.assemble(
            (f"{name} ", f"bold {TOOL}"),
            (self._format_args(args), MUTED),
        )
        self.console.print(
            Padding(
                Panel(
                    Group(header, Rule(style=MUTED), self._format_result(result)),
                    border_style=MUTED,
                    padding=(0, 1),
                ),
                (1, 2, 0, 2),
            )
        )

    @contextmanager
    def working(self, label="thinking"):
        with self.console.status(
            Text(label, style=MUTED), spinner="dots", spinner_style=ACCENT
        ):
            yield

    # ---------------------------------------------------------------- usage

    def usage(self, stats):
        for key, value in stats.items():
            self._totals[key] = self._totals.get(key, 0) + (value or 0)

        parts = " · ".join(
            f"{value:,} {key.replace('_tokens', '')}"
            for key, value in stats.items()
            if value
        )
        self.console.print(Padding(Text(parts, style=MUTED), (1, 0, 0, 2)))

    # -------------------------------------------------------------- helpers

    def _format_args(self, args):
        if len(args) == 1:
            return str(next(iter(args.values())))
        return json.dumps(args)

    def _format_result(self, result):
        lines = result.strip().splitlines() or ["(no output)"]
        shown = lines[:MAX_TOOL_OUTPUT_LINES]
        body = Text("\n".join(shown), style=MUTED)
        hidden = len(lines) - len(shown)
        if hidden > 0:
            body.append(f"\n… {hidden} more lines", style=f"italic {TOOL}")
        return body


ui = UI()

"""Terminal presentation layer.

Knows nothing about LLMs, providers or tools - it only receives plain
strings and dicts and decides how they look. Ported from neuralcode's
ui.py, trimmed to what supacode actually has right now: session
persistence just landed (resumed/replay/pick/note below), but still no
subagents, no write_todos tool, no compaction, no context-reminder
injection. Those methods land here once their features exist - not
before, same rule as ToolSpec.kind.
"""

import json
from contextlib import contextmanager

from rich.console import Console, Group
from rich.live import Live
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

    def pick(self, title, rows):
        """Numbered list; returns the chosen index or None.

        Backs /rewind and /sessions - both need "show me a list, let me
        pick one" and nothing fancier. Any non-digit answer, an
        out-of-range number, or Ctrl-C/EOF all come back as None - the
        caller treats "didn't pick" as "do nothing," not as an error.
        """
        self.console.print(Padding(Text(title, style=f"bold {ACCENT}"), (1, 0, 0, 2)))
        for i, row in enumerate(rows):
            self.console.print(Padding(Text(f"{i:>3}  {row}", style=MUTED), (0, 0, 0, 2)))
        try:
            answer = input("\n  number> ").strip()
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return None
        return int(answer) if answer.isdigit() and int(answer) < len(rows) else None

    # --------------------------------------------------------------- output

    def clear(self):
        self.console.clear()

    def note(self, text):
        self.console.print(Padding(Text(text, style=MUTED), (1, 0, 0, 2)))

    def resumed(self, messages, label="resumed"):
        """One-line header printed right after redrawing a loaded session."""
        turns = sum(1 for m in messages if m["role"] == "user")
        self.console.print(
            Padding(
                Text(f"{label} · {len(messages)} messages · {turns} turns", style=MUTED),
                (0, 0, 0, 2),
            )
        )

    def user(self, text):
        self.console.print(
            Padding(Text(text.strip(), style=f"bold {USER}"), (1, 0, 0, 2))
        )

    def replay(self, messages):
        """Redraw a loaded transcript so the screen matches the history.

        Only needed after /sessions or /rewind clear()s the screen - the
        live path never calls this, since it was already drawn once as
        the turn happened.
        """
        results = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
        for message in messages:
            if message["role"] == "user":
                self.user(message["content"])
            elif message["role"] == "assistant":
                if message.get("content"):
                    self.console.print(self._agent_block(message["content"]))
                for call in message.get("tool_calls") or []:
                    self.tool(call["name"], call["arguments"], results.get(call["id"], ""))

    def _agent_block(self, text):
        return Padding(
            Group(
                Text("agent", style=f"bold {ACCENT}"),
                Padding(Markdown(text.strip()), (1, 0, 0, 0)),
            ),
            (1, 2, 0, 2),
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
    def stream(self):
        """Render the assistant's reply live as call_llm streams it in.

        Replaces working()+agent() for the model-call step: a "thinking"
        placeholder shows until the first token arrives, then the same
        block agent() used to print once now redraws in place as the
        text grows. Yields a callback - call it with the *cumulative*
        text so far, as many times as you like (once per chunk is fine).

        The live region is transient: while streaming it occupies one
        redrawing spot on screen, and on exit it's cleared and the final
        text is printed once, normally, so it scrolls like any other
        output instead of leaving a half-redrawn frame behind. A turn
        that never produces text (pure tool call) leaves nothing printed
        at all - same as the old `if message["content"]: ui.agent(...)`.
        """
        state = {"text": ""}

        def render():
            if not state["text"]:
                return Padding(Text("thinking", style=MUTED), (1, 0, 0, 2))
            return self._agent_block(state["text"])

        with Live(render(), console=self.console, refresh_per_second=12, transient=True) as live:
            def update(text):
                state["text"] = text
                live.update(render())

            yield update

        if state["text"]:
            self.console.print(self._agent_block(state["text"]))

    def approve(self, reason):
        """Block on a y/n prompt for a tool call permissions.check flagged "ask".

        Anything except an explicit yes - "n", garbage input, Ctrl-C/EOF -
        comes back False. A permission prompt has to fail closed: an
        unreadable answer must behave like "no", not like "allow".
        """
        self.console.print(Padding(Text(reason, style=f"bold {TOOL}"), (1, 0, 0, 2)))
        try:
            answer = input("  allow? (y/n)> ").strip()
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return False
        return answer.lower().startswith("y")

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

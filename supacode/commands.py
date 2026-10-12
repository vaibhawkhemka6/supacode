"""Slash commands. Anything typed starting with / lands here.

/rewind and /sessions now exist, backed by session.py's JSONL log.
/compact doesn't yet - it needs compact.py's summarization logic first,
which lands in a later step; until then it falls through to the same
"show available commands" fallback as any unrecognized command.
"""

from . import session
from .ui import ui

COMMANDS = {
    "/rewind": "jump back to an earlier point in this chat",
    "/sessions": "open a past chat",
}


def preview(message):
    if message.get("tool_calls"):
        return "-> " + message["tool_calls"][0]["name"]
    return " ".join(str(message.get("content") or "").split())[:70]


def redraw(messages, label):
    """The screen no longer matches the history, so wipe it and draw again."""
    ui.clear()
    ui.banner()
    ui.resumed(messages, label)
    ui.replay(messages)
    return messages


def rewind(messages):
    rows = [f"{m['role']:<9} {preview(m)}" for m in messages]
    choice = ui.pick("rewind to", rows)
    if choice is None:
        return messages
    session.rewind_to(choice + 1)
    return redraw(messages[: choice + 1], "rewound")


def sessions(messages):
    saved = session.all_sessions()
    if not saved:
        ui.note("no saved chats yet")
        return messages
    rows = [f"{s['id']}  {s['title']}" for s in saved]
    choice = ui.pick("open chat", rows)
    if choice is None:
        return messages
    return redraw(session.open_session(saved[choice]["id"]), "opened")


def handle(command, messages):
    if command == "/rewind":
        return rewind(messages)
    if command == "/sessions":
        return sessions(messages)
    ui.note("\n".join(f"{name}  -  {help}" for name, help in COMMANDS.items()))
    return messages

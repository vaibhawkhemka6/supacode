"""Transcripts on disk. One append-only JSONL file per chat.

Ported from neuralcode's session.py almost unchanged - the only reason it
ports this cleanly is that agent.py's messages are already plain JSON-safe
dicts ({"role", "content", "tool_calls"}), normalized by the provider layer
in models/. NeuralCode has to call message.model_dump(exclude_none=True)
first because it stores raw OpenAI SDK objects; supacode never has that
problem, so save() below can json.dumps() a message straight off the list.

No harness_home/cwd split yet - SESSION_DIR is anchored to Path.cwd(),
same simplification permissions.py's PROJECT already makes. That split is
a later, deliberate Phase 2 (decode) retrofit of this exact path, not
missing by oversight.
"""

import json
from datetime import datetime
from pathlib import Path

PROJECT = str(Path.cwd().resolve()).replace("/", "-")
SESSION_DIR = Path.home() / ".agents" / "sessions" / PROJECT
CURRENT = datetime.now().strftime("%Y%m%d-%H%M%S")
WRITTEN = 0  # how many messages are already flushed to disk


def path_for(session_id):
    return SESSION_DIR / f"{session_id}.jsonl"


def save(messages):
    """Append what's new since the last save. Never rewrite what's on disk.

    Called after every single append to `messages` in agent.py - after the
    assistant message, after every tool result, after a slash command -
    not once at the end of the turn. A crash mid-turn still leaves a
    replayable partial transcript; waiting until "the turn is done" to
    save would lose exactly the turn that crashed.
    """
    global WRITTEN
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    with path_for(CURRENT).open("a") as f:
        for message in messages[WRITTEN:]:
            f.write(json.dumps(message) + "\n")
    WRITTEN = len(messages)


def rewind_to(count):
    """Record a rewind as a new entry - the old messages stay in the file.

    The log is genuinely append-only: /rewind doesn't truncate the file or
    delete anything, it just writes one more line saying "replay should
    stop accumulating at index `count` from here on."
    """
    global WRITTEN
    with path_for(CURRENT).open("a") as f:
        f.write(json.dumps({"rewind_to": count}) + "\n")
    WRITTEN = count


def compacted(messages):
    """Record a compaction's result as a new entry and start counting from it.

    Compaction (Day 13) replaces the whole in-memory list, so the log needs
    a shape that says "discard everything accumulated so far, start from
    exactly this list instead" - same append-only principle as rewind_to,
    just replacing instead of truncating.
    """
    global WRITTEN
    with path_for(CURRENT).open("a") as f:
        f.write(json.dumps({"compacted": messages}) + "\n")
    WRITTEN = len(messages)


def load(session_id):
    """Replay the log top to bottom: messages accumulate, markers rewrite.

    A half-written last line (process killed mid-save) is skipped, not
    raised - all_sessions() below calls load() on every file in the
    directory just to build a picker list, and one corrupt file can't be
    allowed to take the whole /sessions view down with it.
    """
    messages = []
    for line in path_for(session_id).read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "rewind_to" in entry:
            del messages[entry["rewind_to"]:]
        elif "compacted" in entry:
            messages = list(entry["compacted"])
        else:
            messages.append(entry)
    return messages


def open_session(session_id):
    """Switch CURRENT to a past chat, replay it, and become it."""
    global CURRENT, WRITTEN
    CURRENT = session_id
    messages = load(session_id)
    WRITTEN = len(messages)
    return messages


def title(messages):
    for message in messages:
        if message["role"] == "user":
            return " ".join(str(message.get("content") or "").split())[:60]
    return "(empty)"


def all_sessions():
    """Every saved chat for this project, newest first."""
    if not SESSION_DIR.exists():
        return []
    files = sorted(
        SESSION_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return [{"id": p.stem, "title": title(load(p.stem))} for p in files]

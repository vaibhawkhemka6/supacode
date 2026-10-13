"""Keeping tool output from blowing up the context window.

Two tiers live here so far, cheapest first:

1. cap   (9a) - a fresh tool result is trimmed the instant it's created,
           and the full text parked in a temp file the agent can page
           through for the rest of that turn. The decision is made once,
           at creation, so it never has to come back and edit the
           message again.
2. strip (9b) - once a turn is completely over, every tool result it
           produced shrinks to a short stub. Closes the gap cap leaves
           open: several medium results in one turn, none big enough to
           trip cap on its own, that would otherwise just pile up turn
           after turn forever.

Two more tiers land as their own steps: fit/drop (9c - last-resort
trimming inside a single request) and full LLM-summary compaction (9d -
the expensive /compact rebuild). Both are still dead code before
anything calls them.

strip is the first tier that edits messages already sitting in
`messages`, not a fresh string at creation time, so it has to respect the
locked prefix - the frozen system + summary + head that compaction (9d)
will eventually leave behind. `locked()` exists now so strip (and later
fit/drop) never touch it, even though nothing produces a <summary>
marker yet.
"""

import tempfile
from pathlib import Path

CAP = 10_000  # chars of a fresh tool result the agent sees inline
STUB = 300  # chars kept once the turn that produced a tool result is over
TRIMMED = "[output trimmed:"  # marker - lets later tiers tell "already shrunk" from "never touched"
SUMMARY = "<summary>"  # marks the handoff note compaction (9d) will leave behind
SPILLS = []  # temp files belonging to the current turn


def spill(text):
    """Park the full output on disk for the rest of this turn."""
    handle = tempfile.NamedTemporaryFile(
        mode="w", prefix="supacode-", suffix=".txt", delete=False
    )
    handle.write(text)
    handle.close()
    SPILLS.append(Path(handle.name))
    return handle.name


def cap(text):
    """Trim a fresh tool result, leaving a pointer to the whole thing.

    Called right where a tool result is created (bash, read_file) - the
    decision is made once, here, and never revisited. That's what makes
    this the cheapest tier: nothing downstream has to come back and edit
    this message again later.
    """
    if len(text) <= CAP:
        return text

    try:
        path = spill(text)
    except OSError:
        # No temp file (read-only /tmp, no space). Still better to trim
        # and say so than to fail the tool call outright.
        return (
            text[:CAP]
            + f"\n\n{TRIMMED} {len(text) - CAP} chars cut and the rest could not be saved.]"
        )
    return (
        text[:CAP] + f"\n\n{TRIMMED} {len(text) - CAP} of {len(text)} chars cut. "
        f"The whole output is at {path} - page through it with "
        "head, tail, sed -n or grep. It is deleted when this turn ends.]"
    )


def sweep():
    """Delete this turn's temp files. Their paths die with the tool results."""
    for path in SPILLS:
        path.unlink(missing_ok=True)
    SPILLS.clear()


def locked(messages):
    """Length of the frozen prefix - everything up to and including the
    newest <summary> marker a /compact left behind. Derived rather than
    remembered, so it stays correct across /compact, /rewind and switching
    sessions. Nothing produces a <summary> marker yet (that's 9d), so this
    returns 0 for now - but strip (and fit/drop in 9c) call it unconditionally
    so the boundary is already respected once compaction lands.
    """
    for index in range(len(messages) - 1, -1, -1):
        if SUMMARY in (messages[index].get("content") or ""):
            return index + 1
    return 0


def strip(messages):
    """Shrink every tool result that is no longer part of the live turn.

    Called once a turn has finished, so by now "everything unlocked" and
    "everything the model no longer needs in full" are the same set. Unlike
    cap, this edits messages already sent to the model in a prior turn -
    that's why it has to stay behind locked(messages) and never touch the
    cached prefix.
    """
    shrunk = 0
    for message in messages[locked(messages):]:
        content = message.get("content") or ""
        if message["role"] != "tool" or TRIMMED in content or len(content) <= STUB:
            continue

        message["content"] = (
            content[:STUB] + f"\n\n{TRIMMED} {len(content) - STUB} more chars. "
            "Run the command again if you need them.]"
        )
        shrunk += 1
    return shrunk

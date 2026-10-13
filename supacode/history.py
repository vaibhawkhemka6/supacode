"""Keeping tool output from blowing up the context window.

Three tiers live here so far, cheapest first:

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
3. fit   (9c) - runs before every single call to the model, not once per
           tool result and not once per finished turn. Closes the gap
           strip leaves open: a turn that's still in progress, firing
           tool call after tool call, can blow past the window before
           strip ever gets a chance to run. If the running total is over
           budget, fit throws whole tool results away outright, oldest
           first, until it's back under - the harshest tier, used only
           when cap and strip, each doing their job correctly, still
           weren't enough.

One more tier lands as its own step: full LLM-summary compaction (9d -
the expensive /compact rebuild). Still dead code before anything calls
it - but fit's budget already uses the exact threshold compaction will
trigger on, so the two are ready to agree the moment 9d exists.

strip and fit are the two tiers that edit messages already sitting in
`messages`, not a fresh string at creation time like cap, so both have to
respect the locked prefix - the frozen system + summary + head that
compaction (9d) will eventually leave behind. `locked()` exists now so
neither ever touches it, even though nothing produces a <summary> marker
yet.
"""

import json
import tempfile
from pathlib import Path

from . import config

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


def estimate(messages):
    """Rough token count. Good enough to decide whether to panic - not a
    real tokenizer call, just chars / 4 over the whole list."""
    return sum(len(json.dumps(m)) for m in messages) // 4


def fit(messages):
    """Last resort: discard whole tool results, oldest first, until the
    request fits a safety budget - checked before every single call to the
    model, not once per tool result (cap) or once per finished turn (strip).

    Returns how many were dropped. Normally zero - cap and strip already do
    the real work; this only fires when a turn still in progress piles up
    faster than either of them ever gets a chance to act.
    """
    budget = config.CONTEXT_WINDOW * config.COMPACT_AT
    dropped = 0
    for message in messages[locked(messages):]:
        if estimate(messages) <= budget:
            break
        if message["role"] == "tool" and TRIMMED not in (message.get("content") or ""):
            message["content"] = f"{TRIMMED} dropped to fit the context window.]"
            dropped += 1
    return dropped

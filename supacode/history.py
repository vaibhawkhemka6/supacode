"""Keeping tool output from blowing up the context window.

Step 9a: just the first, cheapest tier - cap. Three more tiers land as
their own steps: strip (9b - shrinks a tool result once the turn that
produced it is over), fit/drop (9c - last-resort trimming inside a
single request), and full LLM-summary compaction (9d - the expensive
/compact rebuild). Only cap exists here right now; the other tiers would
be dead code before anything calls them.
"""

import tempfile
from pathlib import Path

CAP = 10_000  # chars of a fresh tool result the agent sees inline
TRIMMED = "[output trimmed:"  # marker - lets later tiers tell "already shrunk" from "never touched"
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

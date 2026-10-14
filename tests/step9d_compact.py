"""Demo / manual-verification script for compact.py (step 9d: the fourth and
final tier of the context cascade - the full LLM-summary rebuild behind
compact.needed()'s automatic trigger in agent.py, and the manual /compact
command in commands.py).

Non-interactive, same reasoning as step9a_cap.py / step9b_strip.py /
step9c_fit.py - no human approval involved. Unlike those three, compact.
summarize() is a real LLM call (compact.py's own dedicated system prompt, no
tools) - monkeypatched out in every case below so this script never hits a
real API and stays deterministic, the same reason step9c_fit.py monkeypatches
config.CONTEXT_WINDOW/COMPACT_AT rather than depending on whatever a real
.env sets them to. Run it directly from the repo root:

    .venv/bin/python3 tests/step9d_compact.py

Two cases, matching the two worked examples in docs/step-9d.md - both
concrete instances of the same structural gap from "## The Problem": fit can
only ever touch role=="tool" messages, so one giant non-tool message is
something NONE of the first three tiers could ever have shrunk, no matter
how many turns went by:

1. huge non-tool-call output  - docs/step-9d.md Example 1. A long, ordinary
                                 transcript, PLUS one single 60,000-char
                                 assistant message that is plain text, not a
                                 tool call. cap/strip/fit all only ever act
                                 on role=="tool", so that one message sat
                                 untouched no matter how many turns went by.
                                 compact() doesn't care about role at all -
                                 it folds that message away into the summary
                                 the moment it falls before the cut point.
                                 Confirms the reassembled shape (system,
                                 summary, tail), confirms summarize() is the
                                 thing that actually read the huge message,
                                 confirms no tool call is ever left orphaned
                                 from its result, and confirms locked() now
                                 returns a real, non-zero index for the
                                 first time.
2. huge user message input    - docs/step-9d.md Example 2. The same gap,
                                 from the other side: this time the oversized
                                 message is something the *user* pasted in,
                                 not something the model wrote. Reproduces
                                 step9c_fit.py's case_2 aftermath almost
                                 verbatim: a <summary>-locked prefix, ten
                                 already-"[output trimmed:...]" tool results,
                                 and one 50,000-char user message fit() was
                                 structurally forbidden from ever touching.
                                 Confirms compact() folds all of it (including
                                 the huge user message) behind a fresh cut
                                 point fit() could never reach.
"""

from supacode import compact, config
from supacode.history import SUMMARY, TRIMMED, locked


def tool_msg(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def assistant_with_call(call_id, name="bash", arguments=None):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "name": name, "arguments": arguments or {"command": "echo hi"}}],
    }


def set_budget(context_window, compact_to, compact_at=0.85):
    """Monkeypatch the three config values compact()'s budget/trigger are
    built from. Returns the (old...) triple to restore."""
    old = (config.CONTEXT_WINDOW, config.COMPACT_TO, config.COMPACT_AT)
    config.CONTEXT_WINDOW = context_window
    config.COMPACT_TO = compact_to
    config.COMPACT_AT = compact_at
    return old


def restore_budget(old):
    config.CONTEXT_WINDOW, config.COMPACT_TO, config.COMPACT_AT = old


def no_orphaned_tool_calls(messages):
    """True if no tool result in `messages` is missing the assistant message
    (with tool_calls) that must sit immediately before it - the one rule
    unique to this tier. cap/strip/fit never need it: none of them ever
    removes a message outright, so none of them can ever leave a tool call
    pointing at a result that no longer exists."""
    for i, m in enumerate(messages):
        if m["role"] == "tool" and not messages[i - 1].get("tool_calls"):
            return False
    return True


def case_1_huge_non_tool_output():
    print("\n--- huge non-tool-call output -> folded into the summary " + "-" * 2)
    # A long, ordinary transcript - dozens of (user, assistant-with-
    # tool_call, tool-result) turns - PLUS one single 60,000-char assistant
    # message that is NOT a tool call, just plain text: docs/step-9d.md
    # Example 1's "model itself wrote one enormous reply... not a tool call,
    # not a tool result." cap/strip/fit all only ever act on role=="tool",
    # so that one message is something none of the first three tiers could
    # ever have reached, no matter how many turns went by.
    messages = [{"role": "system", "content": "sys"}]
    chunk = "x" * 1_200
    for i in range(20):
        messages.append({"role": "user", "content": f"turn {i}: " + chunk})
        messages.append(assistant_with_call(f"call_{i}"))
        messages.append(tool_msg(f"call_{i}", f"result {i}: " + chunk))

    huge_index = len(messages)
    huge_message = {"role": "assistant", "content": "p" * 60_000}  # the 60,000-char plan - plain text, no tool call
    messages.append(huge_message)
    assert "tool_calls" not in huge_message, "the whole point: plain assistant text, not a tool call"

    # Later traffic big enough that it alone crosses COMPACT_TO's budget
    # walking backward - so the cut lands somewhere in HERE, well after
    # huge_index, and the huge message ends up on the summarized side of
    # the cut rather than surviving into the kept tail.
    later_chunk = "x" * 6_000
    for i in range(20):
        messages.append({"role": "user", "content": f"turn {i}: " + later_chunk})
        messages.append(assistant_with_call(f"call_{i}"))
        messages.append(tool_msg(f"call_{i}", f"result {i}: " + later_chunk))

    assert messages[huge_index] is huge_message

    seen = []
    compact.summarize = lambda msgs: (seen.append(msgs), f"<mocked summary over {len(msgs)} messages>")[1]

    old = set_budget(context_window=128_000, compact_to=0.35)
    try:
        before = len(messages)
        kept = compact.compact(messages)
    finally:
        restore_budget(old)

    assert len(kept) < before, "a transcript this big must shrink"
    assert kept[0] == messages[0], "the system message must survive byte-identical"
    assert kept[1]["role"] == "user" and SUMMARY in kept[1]["content"], (
        "index 1 must be the new <summary> handoff message"
    )
    assert not any(m is huge_message for m in kept), (
        "the 60,000-char assistant message that cap/strip/fit could never touch "
        "must be gone now - folded into the summary instead of carried verbatim"
    )
    assert seen and any(m is huge_message for m in seen[0]), (
        "summarize() must be the thing that actually read the huge message"
    )
    assert no_orphaned_tool_calls(kept), "no tool result in the kept tail may be missing its assistant call"
    assert locked(kept) == 2, f"locked() must now see the fresh <summary> at kept[1], got {locked(kept)}"
    print(
        f"confirmed: {before} messages (including the 60,000-char assistant "
        f"message at index {huge_index} that NOTHING in cap/strip/fit could "
        f"ever touch) -> {len(kept)} messages - it survives only inside the "
        f"new <summary> note, and locked(kept) == {locked(kept)}"
    )


def case_2_huge_user_input():
    print("\n--- huge user message input -> folded into the summary " + "-" * 10)
    # The same gap as case 1, from the other side: the oversized message is
    # something the *user* pasted in, not something the model wrote.
    # Reproduces step9c_fit.py's case_2 aftermath almost exactly: an earlier
    # <summary>, ten tool results fit() already fully discarded, and one
    # 50,000-char user message fit() was never allowed to touch (not
    # role=="tool"), plus a long run of later traffic since then.
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "<summary>earlier work summarized here</summary>"},
    ]
    for i in range(10):
        messages.append(tool_msg(f"old_{i}", f"{TRIMMED} dropped to fit the context window.]"))
    huge_index = len(messages)
    huge_message = {"role": "user", "content": "y" * 50_000}  # fit() could never drop this
    messages.append(huge_message)
    later_chunk = "x" * 6_000  # big enough that the later run alone crosses COMPACT_TO's budget
    for i in range(20):
        messages.append({"role": "user", "content": f"later turn {i}: " + later_chunk})
        messages.append(assistant_with_call(f"new_{i}"))
        messages.append(tool_msg(f"new_{i}", f"later result {i}: " + later_chunk))

    assert messages[huge_index] is huge_message

    compact.summarize = lambda msgs: "<mocked summary folding in the 50,000-char message>"

    old = set_budget(context_window=128_000, compact_to=0.35)
    try:
        before = len(messages)
        kept = compact.compact(messages)
    finally:
        restore_budget(old)

    assert len(kept) < before, "compact() must shrink what fit() was stuck with"
    assert not any(m is huge_message for m in kept), (
        "the 50,000-char message fit() was forced to keep forever must be gone "
        "now - folded into the summary instead of carried verbatim"
    )
    assert no_orphaned_tool_calls(kept)
    assert locked(kept) == 2, "a fresh <summary> replaces the old one at the same position"
    print(
        f"confirmed: {before} messages (the old <summary>, fit()'s ten discarded "
        f"stubs, and the 50,000-char message fit() could never touch) -> {len(kept)} "
        f"messages - the huge message folded into the new summary instead of "
        f"carried verbatim"
    )


def main():
    case_1_huge_non_tool_output()
    case_2_huge_user_input()
    print("\nAll step 9d (compact) cases passed.")


if __name__ == "__main__":
    main()

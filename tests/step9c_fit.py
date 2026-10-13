"""Demo / manual-verification script for history.fit (step 9c: the third,
harshest tier of the context cascade - supacode/history.py's fit()/estimate(),
wired into agent.py at the very top of the inner tool-calling loop, before
every single call_llm()).

Non-interactive, same reasoning as step9a_cap.py / step9b_strip.py - fit has
nothing to do with human approval either. Unlike those two, fit doesn't need
any real bash/read_file calls at all: it only ever looks at the shape and
size of `messages`, so every case here builds its `messages` list by hand,
the same shape agent.py builds it, and calls history.fit(messages) directly -
no call_tool(), no ui.tool(). Run it directly from the repo root:

    .venv/bin/python3 tests/step9c_fit.py

config.CONTEXT_WINDOW / config.COMPACT_AT are monkeypatched per case so the
budget is small and exact, instead of depending on whatever a real .env sets
CONTEXT_WINDOW to - the same reason step9a_cap.py's case 3 monkeypatches
tempfile.NamedTemporaryFile rather than relying on a real disk failure.

Two cases, matching the two worked examples in docs/step-9c.md:

1. mid-turn open   - docs/step-9c.md Example 1. Twenty medium tool results
                      in one still-open turn, none of them touched by cap()
                      on its own, that together push estimate(messages) over
                      budget. fit() drops the oldest ones one at a time,
                      re-checking the running total after each drop, and
                      stops the instant it's back under budget - leaving the
                      newest results completely untouched. Confirms both the
                      "oldest first" order and the "stop the moment it fits,
                      don't drop more than needed" behavior.
2. locked + best-  - docs/step-9c.md Example 2. A <summary> marker (the kind
   effort limit        a /compact run would leave behind) locks everything
                      up to and including it; fit() never considers touching
                      that prefix. The unlocked tail has one huge *user*
                      message (role != "tool", so fit() can never drop it)
                      big enough on its own that even discarding every
                      single eligible tool result in the tail still isn't
                      enough to get under budget. Confirms fit() respects
                      locked(), confirms it still drops every eligible
                      message it's allowed to touch, and confirms the
                      documented failure mode: it returns whatever it
                      managed to drop and lets the oversized request go out
                      anyway - fit has no fallback of its own.
"""

from supacode import config, history


def tool_msg(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def set_budget(context_window, compact_at):
    """Monkeypatch the two config values fit()'s budget is built from.
    Returns the (old_context_window, old_compact_at) pair to restore."""
    old = (config.CONTEXT_WINDOW, config.COMPACT_AT)
    config.CONTEXT_WINDOW = context_window
    config.COMPACT_AT = compact_at
    return old


def restore_budget(old):
    config.CONTEXT_WINDOW, config.COMPACT_AT = old


def case_1_mid_turn_still_open():
    print("\n--- mid-turn still open -> partial, oldest-first drop " + "-" * 2)
    # Eighteen tool results from one still-open turn, each ~4,000 chars -
    # the exact shape of docs/step-9c.md Example 1: "eighteen tool calls
    # into one turn so far, each a medium ~4,000 chars - every one under
    # cap's 10,000, so cap never fired." The doc's run drops exactly 2 of
    # the 18 (the oldest) before it's back under budget; this reproduces
    # that same 2-of-18 outcome against a real, computed budget.
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "run the big batch job"},
    ]
    for i in range(18):
        messages.append(tool_msg(f"call_{i}", f"result #{i}: " + ("x" * 4_000)))

    def estimate_after_dropping(k):
        """What estimate(messages) would be if the first k tool results
        (oldest first) were already dropped - used only to pick an exact
        budget below, never to decide what the real fit() call does."""
        simulated = messages[:2] + [
            {**m, "content": f"{history.TRIMMED} dropped to fit the context window.]"}
            if i < k else m
            for i, m in enumerate(messages[2:])
        ]
        return history.estimate(simulated)

    full_estimate = estimate_after_dropping(0)
    target_k = 2  # pick a budget that needs exactly 2 drops, out of 18 - matches the doc's own run
    budget = estimate_after_dropping(target_k)
    assert estimate_after_dropping(target_k - 1) > budget, (
        "budget picked wrong: dropping one fewer should still be over budget"
    )
    assert full_estimate > budget, "sanity check: the untouched turn must start over budget"

    old = set_budget(context_window=budget, compact_at=1.0)
    try:
        before_untouched = [dict(m) for m in messages[2 + target_k:]]
        dropped = history.fit(messages)
    finally:
        restore_budget(old)

    assert dropped == target_k, f"expected exactly {target_k} drops, got {dropped}"
    for i in range(target_k):
        # Full discard, not a trim like strip()'s 300-char stub - the
        # content becomes exactly this one fixed sentence, nothing else.
        assert messages[2 + i]["content"] == f"{history.TRIMMED} dropped to fit the context window.]", (
            f"tool result #{i} should have been fully discarded, not partially trimmed"
        )
    for i, original in enumerate(before_untouched):
        assert messages[2 + target_k + i] == original, (
            f"tool result #{target_k + i} should be byte-identical - fit() must stop the instant it fits"
        )
    final_estimate = history.estimate(messages)
    assert final_estimate <= budget, f"final estimate {final_estimate} should now be <= budget {budget}"
    print(
        f"confirmed: started at {full_estimate} (budget {budget}), dropped exactly "
        f"{dropped} of 18 oldest-first, ended at {final_estimate} - the other "
        f"{18 - dropped} results are completely untouched"
    )


def case_2_locked_and_best_effort_limit():
    print("\n--- locked boundary + best-effort limit " + "-" * 10)
    # A <summary> marker locks everything up to and including it - the
    # frozen prefix a /compact run would leave behind (step 9d). Matches
    # docs/step-9c.md Example 2 exactly: "<summary> sits at index 2, so
    # locked(messages) == 3."
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "earlier conversation, before the last /compact"},
        {"role": "assistant", "content": "<summary>earlier work summarized here</summary>"},
    ]
    locked_len = history.locked(messages)
    assert locked_len == 3, f"expected the <summary> at index 2 to lock the first 3 messages, got {locked_len}"

    # Unlocked tail: five normal tool results, ONE huge non-tool (user)
    # message fit() can never drop, then five more tool results.
    for i in range(5):
        messages.append(tool_msg(f"call_a{i}", f"result a{i}: " + ("x" * 500)))
    messages.append({"role": "user", "content": "y" * 50_000})  # fit() can't touch this - not role=="tool"
    for i in range(5):
        messages.append(tool_msg(f"call_b{i}", f"result b{i}: " + ("x" * 500)))

    locked_prefix_before = [dict(m) for m in messages[:locked_len]]
    huge_index = next(i for i, m in enumerate(messages) if m["role"] == "user" and len(m["content"]) == 50_000)
    huge_message_before = dict(messages[huge_index])
    eligible_tool_count = sum(1 for m in messages[locked_len:] if m["role"] == "tool")  # computed BEFORE fit() runs
    assert eligible_tool_count == 10, f"expected 10 unlocked tool results set up, got {eligible_tool_count}"

    # Budget so low that even dropping every eligible tool result in the
    # unlocked tail still leaves the huge user message's own size over it.
    old = set_budget(context_window=1, compact_at=0.001)
    try:
        dropped = history.fit(messages)
    finally:
        restore_budget(old)

    budget = 1 * 0.001

    assert dropped == eligible_tool_count, (
        f"expected every one of the {eligible_tool_count} unlocked tool results dropped, got {dropped}"
    )
    for m in messages[locked_len:]:
        if m["role"] == "tool":
            # Full discard, not a trim - the exact fixed sentence, nothing else.
            assert m["content"] == f"{history.TRIMMED} dropped to fit the context window.]", (
                "every unlocked tool result should have been fully discarded"
            )
    assert messages[:locked_len] == locked_prefix_before, "fit() must never touch the locked prefix"
    assert messages[huge_index] == huge_message_before, "fit() must never touch a non-tool message, no matter how big"

    final_estimate = history.estimate(messages)
    assert final_estimate > budget, (
        "this is the documented failure mode: even dropping everything eligible "
        "still isn't enough, so the request goes out over budget anyway"
    )
    print(
        f"confirmed: locked prefix (first {locked_len} messages) untouched, "
        f"all {dropped} eligible tool results dropped, the one huge non-tool "
        f"message stayed at {len(huge_message_before['content'])} chars untouched, "
        f"and estimate {final_estimate} is still > budget {budget} - fit() has no fallback of its own"
    )


def main():
    case_1_mid_turn_still_open()
    case_2_locked_and_best_effort_limit()
    print("\nAll step 9c (fit) cases passed.")


if __name__ == "__main__":
    main()

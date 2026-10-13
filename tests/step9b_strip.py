"""Demo / manual-verification script for history.strip (step 9b: the second
tier of the context cascade - supacode/history.py's strip()/locked(), wired
into agent.py right after history.sweep() once a turn is completely over).

Non-interactive, same reasoning as step9a_cap.py - strip has nothing to do
with human approval either, so every bash/read_file call below is deliberately
picked to land on an "allow" rule in permissions.BASH_RULES (cat, pwd) or is
unconditionally allowed (read_file), and it runs start to finish with no y/n
prompt. Run it directly from the repo root (cwd matters - PROJECT is
Path.cwd().resolve() in both permissions.py and sandbox.py):

    .venv/bin/python3 tests/step9b_strip.py

Unlike step9a_cap.py, this script doesn't run through agent.py's full main()
loop (that's interactive - it reads real stdin for user turns), so each case
builds its own `messages` list by hand, the same shape agent.py builds it
(```{"role": "tool", "tool_call_id": ..., "content": result}```), then calls
history.sweep() and history.strip(messages) in that exact order - the same
order and cadence agent.py uses once a turn's inner tool-calling loop breaks.

Three cases, matching the three worked examples in docs/step-9b.md. All
three happen inside ONE turn in the doc (re-read it: "Same turn also had
bash..." and "Earlier in the same turn, read_file...") so they're exercised
together here as one turn with three tool calls, then one sweep()+strip()
pass - not three separate turns:

1. medium result  - bash("cat <2,400-char file>"). Under CAP (10,000), so
                     cap() left it completely untouched at creation. strip()
                     is the first thing to ever touch it: shrinks to a
                     300-char (STUB) stub.
2. tiny result     - bash("pwd"). Comfortably under STUB (300) on its own,
                     so strip() skips it - nothing to save by shrinking an
                     already-tiny message.
3. already-capped  - read_file() on an 85,000-char file. cap() already
                     trimmed this to ~10,100 chars + the shared TRIMMED
                     marker when it was created. strip() sees TRIMMED
                     already present and skips it completely - it stays at
                     ~10,100 chars forever, not 300. The quirk: sweep() runs
                     BEFORE strip(), so the spill path still embedded in this
                     message's text is already deleted from disk by the time
                     strip() even looks at it.

history.locked() exists and is already wired into strip()'s loop (see
history.py), but there's no case here exercising the locked-boundary itself -
nothing in supacode can produce a real "<summary>" marker yet (that needs
compaction, step 9d), and docs/step-9b.md deliberately stops at these three
examples for the same reason. That worked walkthrough, against a real
/compact run instead of a constructed one, lands in step 9d's own test.
"""

from pathlib import Path

from supacode import history
from supacode.tools import call_tool
from supacode.ui import ui

MEDIUM_FILE = Path(__file__).parent / "_step9b_demo_medium_file.txt"
HUGE_FILE = Path(__file__).parent / "_step9b_demo_huge_file.txt"


def run(label, name, args):
    print(f"\n--- {label} " + "-" * (50 - len(label)))
    result = call_tool(name, args)
    ui.tool(name, args, result)
    return result


def case_1_2_3_one_turn_three_results():
    # One turn, three tool calls, mirroring docs/step-9b.md Examples 1-3.
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "run the tests, check pwd, and read the lockfile"},
    ]

    # --- tool call 1: medium result, well under CAP, cap() never touches it
    medium_text = "PASSED test_thing\n" * 133  # 2,527 chars - over STUB, under CAP
    MEDIUM_FILE.write_text(medium_text)
    result_1 = run("medium result -> cap() no-op", "bash", {"command": f"cat {MEDIUM_FILE}"})
    assert history.TRIMMED not in result_1, "medium result should be untouched by cap()"
    assert len(result_1) > history.STUB, "medium result must be big enough for strip() to act on"
    messages.append({"role": "tool", "tool_call_id": "call_1", "content": result_1})

    # --- tool call 2: tiny result, comfortably under STUB on its own
    result_2 = run("tiny result -> small on its own", "bash", {"command": "pwd"})
    assert len(result_2) <= history.STUB, "pwd output should already be <= STUB"
    messages.append({"role": "tool", "tool_call_id": "call_2", "content": result_2})

    # --- tool call 3: huge result, cap() already capped + spilled it at creation
    huge_text = "lockfile line\n" * 7_000  # 98,000 chars - well over CAP
    HUGE_FILE.write_text(huge_text)
    result_3 = run("huge result -> already capped by cap()", "read_file", {"path": str(HUGE_FILE)})
    assert history.TRIMMED in result_3, "huge result should already carry cap()'s TRIMMED marker"
    assert len(history.SPILLS) == 1, f"expected exactly one spill (from call 3), got {history.SPILLS}"
    spill_path = history.SPILLS[0]
    assert spill_path.exists(), "spill file should still exist while the turn is open"
    messages.append({"role": "tool", "tool_call_id": "call_3", "content": result_3})
    capped_len = len(result_3)

    # --- turn ends: agent.py's exact order, sweep() then strip()
    history.sweep()
    assert not spill_path.exists(), "sweep() should have deleted the spill file"
    shrunk = history.strip(messages)

    # Example 1: medium result got shrunk to the STUB.
    stripped_1 = messages[2]["content"]
    assert history.TRIMMED in stripped_1, "medium result should now carry strip()'s TRIMMED marker"
    assert len(stripped_1) <= history.STUB + 100, f"medium result should be stubbed, got {len(stripped_1)} chars"
    assert stripped_1.startswith(result_1[: history.STUB]), "the stub should be a prefix of the original"
    print(f"confirmed: medium result ({len(result_1)} chars) shrunk to a {len(stripped_1)}-char stub")

    # Example 2: tiny result is byte-identical, strip() never touched it.
    untouched_2 = messages[3]["content"]
    assert untouched_2 == result_2, "tiny result should be completely untouched by strip()"
    print(f"confirmed: tiny result ({len(result_2)} chars) left completely untouched")

    # Example 3: already-capped result is untouched too - but for a different
    # reason (shared TRIMMED marker short-circuits strip(), not its size) -
    # and it stays at cap()'s ~10,100 chars, NOT shrunk to strip()'s 300.
    untouched_3 = messages[4]["content"]
    assert untouched_3 == result_3, "already-capped result should be byte-identical after strip()"
    assert len(untouched_3) > history.STUB, "the quirk: strip() does NOT chase the extra savings here"
    assert str(spill_path) in untouched_3, (
        "the message still names the now-deleted spill path - nothing in the "
        "text itself flags that sweep() already deleted it"
    )
    print(
        f"confirmed: already-capped result stays at {len(untouched_3)} chars (not {history.STUB}), "
        f"still pointing at the now-deleted {spill_path}"
    )

    assert shrunk == 1, f"exactly one message (the medium one) should have been shrunk this turn, got {shrunk}"
    print(f"confirmed: strip() shrunk exactly {shrunk} of 3 tool results this turn")

    MEDIUM_FILE.unlink(missing_ok=True)
    HUGE_FILE.unlink(missing_ok=True)


def main():
    case_1_2_3_one_turn_three_results()
    print("\nAll step 9b (strip) cases passed.")


if __name__ == "__main__":
    main()

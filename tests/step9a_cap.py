"""Demo / manual-verification script for history.cap (step 9a: the first,
cheapest tier of the context cascade - supacode/history.py's cap(), wired
into bash()/read_file() in tools/registry.py, swept by agent.py).

Non-interactive, unlike step6_permissions.py and step7_sandbox.py - cap
has nothing to do with human approval, it's pure size-based trimming, so
every bash/read_file call below is deliberately picked to land on an
"allow" rule in permissions.BASH_RULES (cat, grep) so the script runs
start to finish with no y/n prompt. Run it directly from the repo root
(cwd matters - PROJECT is Path.cwd().resolve() in both permissions.py and
sandbox.py):

    .venv/bin/python3 tests/step9a_cap.py

Three cases, matching the three examples in docs/step-9a.md:
1. small output  - cap() is a no-op, nothing spilled, nothing to sweep.
2. big output    - capped + spilled, the model gets a working pointer
                    and reuses it in the same turn via a follow-up grep,
                    then sweep() deletes the spill file once the turn
                    "ends" (sweep() is called by hand here, same as
                    agent.py calls it once per completed turn).
3. spill failure - tempfile.NamedTemporaryFile monkeypatched to raise
                    OSError, so cap() falls back to "trimmed, and the
                    rest could not be saved" with no path and no SPILLS
                    entry. This one calls history.cap() directly rather
                    than through a tool, since it's a disk-failure
                    condition, not something a real command produces.
"""

import tempfile
from pathlib import Path

from supacode import history
from supacode.tools import call_tool
from supacode.ui import ui

DEMO_FILE = Path(__file__).parent / "_step9a_demo_big_file.txt"


def run(label, name, args):
    print(f"\n--- {label} " + "-" * (50 - len(label)))
    result = call_tool(name, args)
    ui.tool(name, args, result)
    return result


def case_1_small_output():
    # "ls" and "echo" are both "allow" in BASH_RULES - no prompt. 420-ish
    # chars back, nowhere near CAP (10,000), so cap() hands it back
    # unchanged. Nothing spilled, SPILLS stays empty.
    result = run("small output -> no-op", "bash", {"command": "echo hello from cap demo"})

    assert history.TRIMMED not in result, "small output should pass through untouched"
    assert history.SPILLS == [], "nothing should be spilled for small output"
    print("confirmed: untouched, SPILLS == []")


def case_2_big_output_capped_and_reused():
    # Build a file > CAP chars with a needle planted past the first
    # 10,000 chars, so the model (a human, here) genuinely needs the
    # spill pointer to find it - mirrors docs/step-9a.md Example 2.
    # One short line per entry (not one giant unbroken line) - that way
    # the follow-up grep's match is just "6001:FINDME_42", a few bytes,
    # not the entire file re-joined into one line that's itself over
    # CAP and gets capped a second time, hiding the needle again.
    needle = "FINDME_42"
    content = ("a\n" * 5_000) + f"{needle}\n" + ("b\n" * 1_000)
    DEMO_FILE.write_text(content)

    try:
        # read_file is unconditionally "allow" - no prompt.
        result = run("big output -> capped + spilled", "read_file", {"path": str(DEMO_FILE)})

        assert history.TRIMMED in result, "output over CAP chars should be trimmed"
        assert len(history.SPILLS) == 1, f"expected exactly one spill, got {history.SPILLS}"
        spill_path = history.SPILLS[0]
        assert spill_path.exists(), "spill file should exist while the turn is still open"
        assert spill_path.read_text() == content, "spill file should hold the FULL original text"
        print(f"confirmed: spilled to {spill_path}, full content recoverable from disk")

        # The model reuses the pointer in the same turn - "grep *" is
        # "allow", so this is the follow-up tool call from Example 2,
        # not a human typing it by hand.
        result = run(
            "follow-up tool call reuses the spill file",
            "bash",
            {"command": f"grep -n {needle} {spill_path}"},
        )
        assert needle in result, "grep against the spill file should find the needle cap() couldn't show inline"
        print("confirmed: the needle past char 10,000 was only reachable via the spill pointer")

    finally:
        # agent.py calls this once per completed turn, win or not -
        # simulate "the turn just ended."
        history.sweep()
        assert history.SPILLS == [], "sweep() should clear SPILLS"
        assert not spill_path.exists(), "sweep() should delete the spill file from disk"
        print(f"confirmed: sweep() deleted {spill_path} and cleared SPILLS")
        DEMO_FILE.unlink(missing_ok=True)


def case_3_spill_failure_fallback():
    # No tool call here - this is a disk-failure condition (read-only or
    # full /tmp), not something `cat` or `python3` can reliably produce
    # on demand, so it's exercised directly against history.cap().
    real_named_temp_file = tempfile.NamedTemporaryFile

    def broken_named_temp_file(*args, **kwargs):
        raise OSError("no space left on device (simulated)")

    tempfile.NamedTemporaryFile = broken_named_temp_file
    try:
        text = "z" * 20_000
        result = history.cap(text)
    finally:
        tempfile.NamedTemporaryFile = real_named_temp_file

    print(f"\n--- spill fails (no writable /tmp) " + "-" * 14)
    print(result[-150:])

    assert result.startswith("z" * history.CAP), "the first CAP chars should still come through"
    assert "could not be saved" in result, "fallback message should say so plainly"
    assert "/tmp" not in result.split(history.TRIMMED)[-1] or "page through" not in result, (
        "fallback should NOT offer a path to page through - there isn't one"
    )
    assert history.SPILLS == [], "a failed spill must never register in SPILLS"
    print("confirmed: trimmed with no path, nothing added to SPILLS")


def main():
    case_1_small_output()
    case_2_big_output_capped_and_reused()
    case_3_spill_failure_fallback()
    print("\nAll step 9a (cap) cases passed.")


if __name__ == "__main__":
    main()

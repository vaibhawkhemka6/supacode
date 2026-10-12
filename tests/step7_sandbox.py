"""Demo / manual-verification script for the sandbox (step 7/8:
supacode/tools/sandbox.py, wired into bash/write_file/str_replace in
registry.py).

Interactive, like tests/step6_permissions.py - no monkeypatching. The
point of this script isn't "does the tool run," it's "does the thing
permissions.py can't guarantee actually hold." Run it directly:

    .venv/bin/python3 tests/step7_sandbox.py

Two cases:
1. write_file outside the project - ASK, then answer "y". Notice the
   write still fails. permissions.check already said "ask" here (same
   as step 6); the difference this step adds is that saying yes no
   longer matters - confine() raises before any bytes touch disk,
   regardless of what the human just approved.
2. bash with a command that would normally be free to exhaust the
   host - allowed to run (it's just `python3 -c ...` - benign-looking,
   no BASH_RULES pattern denies it, so it either auto-allows or asks),
   but it dies on resource.RLIMIT_CPU/RLIMIT_NPROC from inside
   run_sandboxed, not on the 120s wall-clock timeout bash() already
   had and not on a human noticing a runaway process in time.
"""

import tempfile
from pathlib import Path

from supacode.tools import call_tool
from supacode.ui import ui


def run(label, name, args):
    print(f"\n--- {label} " + "-" * (50 - len(label)))
    result = call_tool(name, args)
    ui.tool(name, args, result)
    return result


def main():
    # 1. write_file outside the project: same "ask" verdict as step 6,
    #    but answering "y" now hits a wall sandbox.confine() put there,
    #    not the filesystem succeeding because a human said so.
    outside = Path(tempfile.gettempdir()) / "_demo_sandbox_outside.txt"
    result = run(
        "write_file outside project: ASK -> type y, watch it still fail",
        "write_file",
        {"path": str(outside), "content": "this should never land\n"},
    )
    assert not outside.exists(), (
        "sandbox gap: file landed outside the project despite confine()"
    )
    print(f"\nconfirmed: {outside} does not exist, whether you typed y or n.")
    print(f"result the model would have seen: {result!r}")

    # 2. bash: a CPU-bound loop that permissions.py has no opinion on
    #    strong enough to deny - it isn't rm, sudo, curl, anything on
    #    the deny list - so it either runs or asks, same as any other
    #    unclassified command. What's different this step: once it
    #    runs, resource.RLIMIT_CPU (30s, set in sandbox.py) kills it
    #    from underneath, well before the 120s wall-clock timeout
    #    bash() already enforced would have even noticed.
    run(
        "bash: CPU-bound loop -> dies on RLIMIT_CPU, not the 120s timeout",
        "bash",
        {"command": "python3 -c \"while True: pass\""},
    )


if __name__ == "__main__":
    main()

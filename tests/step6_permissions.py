"""Demo / manual-verification script for the permission gate (step 6/7:
permissions.py + the call_tool gate + ui.approve).

Interactive - no monkeypatching here. Every "ask" case below calls the
real call_tool -> permissions.check -> ui.approve -> input() path, so you
get an actual y/n prompt on your terminal and answer it yourself. Run it
directly:

    .venv/bin/python3 tests/step6_permissions.py

Three cases, one each:
1. bash  - ASK   (no BASH_RULES pattern matches -> catch-all "ask")
2. bash  - DENY  (matches "rm *" -> blocked outright, you will NOT be
                   prompted at all - that's the thing to notice)
3. write_file - ASK (path outside the project directory)

Answer "y" to see the action actually run; answer "n" (or anything else,
or Ctrl-C) to see it denied - either way the model/you get a result
string back, never a crash.
"""

import tempfile
from pathlib import Path

from supacode.tools import call_tool
from supacode.ui import ui


def run(label, name, args):
    print(f"\n--- {label} " + "-" * (50 - len(label)))
    result = call_tool(name, args)
    ui.tool(name, args, result)


def main():
    # 1. bash - ask: nothing in BASH_RULES matches "npm --version", so it
    #    falls to the catch-all "*": "ask". You'll be prompted for real.
    run("bash: ASK -> type y or n", "bash", {"command": "npm --version"})

    # 2. bash - deny: matches "rm *" -> blocked by policy before ui.approve
    #    is ever called. No prompt should appear for this one.
    run("bash: DENY -> no prompt expected", "bash", {"command": "rm -rf /tmp/wont-run"})

    # 3. write_file - ask: path resolves outside the project directory.
    #    You'll be prompted for real again.
    outside = Path(tempfile.gettempdir()) / "_demo_permissions_outside.txt"
    run("write_file: ASK -> type y or n", "write_file",
        {"path": str(outside), "content": "written outside\n"})

    # cleanup - only exists if you answered "y" above
    outside.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

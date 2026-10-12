"""Which tool calls need a human.

sandbox.py decides what is *possible* - confined paths, capped subprocess
resources, enforced from inside the tool functions themselves, so it
holds even if this module is bypassed entirely. This module only decides
what is worth interrupting a human for - read-only or clearly-safe calls
run silently, genuinely destructive ones never run no matter who says
yes, and everything in between stops and asks. The two are deliberately
separate: this is a cooperative convention one disciplined call site
(call_tool) enforces, not a physical guarantee - sandbox.py is where the
physical guarantee actually lives.

Three verdicts, and deny beats ask beats allow:
- "allow": run it, no interruption.
- "ask":   show the human a reason, block on y/n. A "no" (or Ctrl-C) is
           treated exactly like "deny" - the tool never runs.
- "deny":  never run, full stop. Not even offered to the human - some
           actions are too easy to rubber-stamp by habit (rm -rf, git
           push, sudo) to leave the decision to an "are you sure?" prompt.
"""

from fnmatch import fnmatch
from pathlib import Path

PROJECT = Path.cwd().resolve()

# Last matching rule wins is NOT how this works - every matching rule is
# collected and decide() takes the strictest one. So order here doesn't
# matter for correctness, only for readability (catch-all first, then
# specific overrides).
BASH_RULES = {
    "*": "ask",
    # read-only: let them through without interrupting
    "ls*": "allow",
    "pwd": "allow",
    "cd *": "allow",
    "echo *": "allow",
    "sort*": "allow",
    "uniq*": "allow",
    "cut *": "allow",
    "basename *": "allow",
    "dirname *": "allow",
    "date*": "allow",
    "env": "allow",
    "cat *": "allow",
    "head *": "allow",
    "tail *": "allow",
    "wc *": "allow",
    "file *": "allow",
    "which *": "allow",
    "grep *": "allow",
    "rg *": "allow",
    "find *": "allow",
    "tree*": "allow",
    "git status*": "allow",
    "git diff*": "allow",
    "git log*": "allow",
    "git show*": "allow",
    "git ls-files*": "allow",
    "pytest*": "allow",
    "python -m pytest*": "allow",
    # risky: never, even if the human says yes
    "rm *": "deny",
    "sudo *": "deny",
    "chmod *": "deny",
    "chown *": "deny",
    "curl *": "deny",
    "wget *": "deny",
    "git push*": "deny",
    "git reset*": "deny",
    "git clean*": "deny",
}


def split_command(command):
    """Split a compound command on the separators that actually separate.

    A naive split on | and ; also cuts inside quotes, so `rg "cap|max"`
    breaks into fragments that match no rule and fall through to "ask".
    Anything quoted or backslash-escaped is an argument, not a separator.
    """
    parts, current, quote = [], [], None
    index = 0
    while index < len(command):
        char = command[index]
        if quote:
            current.append(char)
            quote = None if char == quote else quote
        elif char == "\\":
            current.append(char)
            index += 1
            if index < len(command):
                current.append(command[index])
        elif char in "\"'":
            quote = char
            current.append(char)
        elif char in "&|;":
            parts.append("".join(current))
            current = []
            while index + 1 < len(command) and command[index + 1] in "&|":
                index += 1
        else:
            current.append(char)
        index += 1

    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def decide(command):
    """Rate every part of a compound command; the strictest verdict wins.

    "ls && rm -rf /" must come back "deny" even though "ls" alone is
    "allow" - letting the strictest part decide is what stops a denied
    command from riding into the session behind an allowed one.
    """
    verdicts = []
    for part in split_command(command):
        action = "ask"
        for pattern, rule in BASH_RULES.items():
            if fnmatch(part, pattern):
                action = rule
        verdicts.append(action)

    for strictest in ("deny", "ask"):
        if strictest in verdicts:
            return strictest
    return "allow"


def inside_project(path):
    return PROJECT in Path(path).resolve().parents


def check(name, args):
    """Return (action, reason). Action is allow, ask, or deny.

    bash gets the rule-table treatment above. write_file/str_replace get a
    narrower check: there's no deny tier for file tools, because the
    destination path is always a legible reason a human can decide on -
    "ask" is enough, nothing needs to be forbidden outright. Every other
    tool (read_file, for now) has no side effects worth gating, so it's
    unconditionally "allow".
    """
    if name == "bash":
        return decide(args["command"]), f"run: {args['command']}"

    if name in ("write_file", "str_replace") and not inside_project(args["path"]):
        return "ask", f"{name} outside {PROJECT}: {args['path']}"

    return "allow", None

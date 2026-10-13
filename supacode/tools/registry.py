"""Flat tool registry.

Every tool the agent can call is declared here once, as a ToolSpec.
`bash`, `read_file`, `write_file`, and `str_replace` are wired up, enough
to prove tool-calling works end to end before write_todos joins this same
list (step 5).

call_tool now runs every call through permissions.check (step 7) before
touching the tool's func - deny blocks it outright, ask blocks on a
human y/n via ui.approve, allow runs it straight through. The tool
functions themselves stay ignorant of all this; they're exactly the bare
functions they were before the gate existed.

No ToolKind yet on purpose: that classification (read_only / file_edit /
other) only earns its keep once something other than permissions.check's
own name-based dispatch needs to read it. Adding it now would be a field
nothing consumes.

Sandbox (step 8) now sits underneath bash/write_file/str_replace, called
directly from inside these functions rather than from call_tool - so the
physical limits (confined paths, capped subprocess resources) hold even
if something calls these functions straight, skipping the permission
gate entirely. The functions' shape doesn't change: same arguments, same
"always return a string, never raise" contract. read_file stays bare on
purpose - it was already unconditionally allowed, and confining it too
would block the agent from inspecting anything outside its own project.

history.cap (step 9a) now wraps bash's and read_file's return values -
the two tools whose output size is unbounded. write_file and str_replace
stay unwrapped: they only ever return a short confirmation string, never
the file content itself, so there's nothing for cap to do there.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import history
from . import sandbox


@dataclass(frozen=True)
class ToolSpec:
    name: str
    func: Callable[..., str]
    description: str
    parameters: dict  # plain JSON schema; each provider adapts it to its own wire format


def bash(command: str) -> str:
    """Run a shell command and return its combined stdout and stderr.

    Runs through sandbox.run_sandboxed, not a raw subprocess.run - the
    wall-clock timeout below is unchanged, but the subprocess tree now
    also carries capped CPU time, process count, and (where supported)
    address space, so a forkbomb or a memory leak dies on its own limits
    instead of relying on the timeout (or a human) to notice in time.
    """
    try:
        result = sandbox.run_sandboxed(command, timeout=120)
    except subprocess.TimeoutExpired as expired:
        # Hand the failure back as a result, not an exception - a slow
        # command is the model's problem to work around, not a reason to
        # take the process down.
        return (
            f"Timed out after {expired.timeout}s and was killed. "
            "Narrow it down rather than re-running the same thing."
        )
    return history.cap((result.stdout + result.stderr) or "(no output)")


def read_file(path: str) -> str:
    """Read a file and return its contents."""
    try:
        return history.cap(Path(path).read_text())
    except OSError as error:
        return f"Error reading {path}: {error}"
    except UnicodeDecodeError as error:
        # read_text() assumes UTF-8. A binary file (image, .pyc, compiled
        # artifact, ...) raises UnicodeDecodeError, which is a ValueError,
        # not an OSError - it wouldn't be caught above. Hand it back as a
        # result instead of letting it propagate and kill the whole agent
        # loop (call_tool has no containment of its own).
        return f"Can't read {path} as text: {error}. It looks like a binary file."


def write_file(path: str, content: str) -> str:
    """Write content to a file, creating it (and overwriting it) as needed.

    confine() resolves path first and raises if it lands outside the
    project - this holds regardless of what permissions.check decided
    upstream, so a human mistakenly saying "y" to a write outside the
    project still can't make it land there.
    """
    try:
        resolved = sandbox.confine(path)
    except sandbox.SandboxError as error:
        return f"Blocked by sandbox: {error}"
    try:
        resolved.write_text(content)
    except OSError as error:
        return f"Error writing {path}: {error}"
    return f"Wrote {len(content)} characters to {path}"


def str_replace(path: str, old_str: str, new_str: str) -> str:
    """Replace one exact-match occurrence of old_str with new_str in a file."""
    try:
        resolved = sandbox.confine(path)
    except sandbox.SandboxError as error:
        return f"Blocked by sandbox: {error}"
    try:
        text = resolved.read_text()
    except OSError as error:
        return f"Error reading {path}: {error}"
    except UnicodeDecodeError as error:
        return f"Can't read {path} as text: {error}. It looks like a binary file."

    count = text.count(old_str)
    if count == 0:
        return f"No match found for old_str in {path}"
    if count > 1:
        return f"old_str is not unique in {path} ({count} matches) - include more surrounding context"

    try:
        resolved.write_text(text.replace(old_str, new_str, 1))
    except OSError as error:
        return f"Error writing {path}: {error}"
    return f"Replaced 1 occurrence in {path}"


TOOLS = [
    ToolSpec(
        name="bash",
        func=bash,
        description="Run a shell command and return its combined stdout and stderr.",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run",
                }
            },
            "required": ["command"],
        },
    ),
    ToolSpec(
        name="read_file",
        func=read_file,
        description="Read a file and return its contents.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                }
            },
            "required": ["path"],
        },
    ),
    ToolSpec(
        name="write_file",
        func=write_file,
        description="Write content to a file, creating it or overwriting it if it already exists.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
    ),
    ToolSpec(
        name="str_replace",
        func=str_replace,
        description=(
            "Replace an exact string match in a file with new text. "
            "old_str must match exactly once in the file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit",
                },
                "old_str": {
                    "type": "string",
                    "description": "Exact text to find (must be unique in the file)",
                },
                "new_str": {
                    "type": "string",
                    "description": "Text to replace it with",
                },
            },
            "required": ["path", "old_str", "new_str"],
        },
    ),
]

TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def call_tool(name: str, args: dict) -> str:
    """Run a tool call through the permission gate, then the tool itself.

    Imports are local (not at module top) to dodge a cycle: ui.py doesn't
    import tools today, but the registry is the thing every future entry
    point calls into, so keeping the dependency direction one-way here -
    tools reaches out to permissions/ui, neither reaches back in - is
    cheap insurance against that becoming circular later.

    No error containment yet (step 4's inner loop adds that) - this
    exists so a smoke test can prove a tool call, gated, round-trips end
    to end.
    """
    from . import permissions
    from ..ui import ui

    action, reason = permissions.check(name, args)
    if action == "deny":
        return f"Blocked by policy: {reason}"
    if action == "ask" and not ui.approve(reason):
        return "The user denied this tool call."

    return TOOLS_BY_NAME[name].func(**args)

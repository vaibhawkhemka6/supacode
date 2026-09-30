"""Flat tool registry.

Every tool the agent can call is declared here once, as a ToolSpec.
`bash`, `read_file`, `write_file`, and `str_replace` are wired up, enough
to prove tool-calling works end to end before write_todos joins this same
list (step 5).

No ToolKind yet on purpose: that classification (read_only / file_edit /
other) only earns its keep once the permission gate (step 6) exists to
read it. Adding it now would be a field nothing consumes. It goes back on
ToolSpec in that step, not before.

No sandbox (step 7), no permission check (step 6), no output capping
(step 8) yet - these run bare, straight against the real filesystem /
a raw subprocess. Those wrap these same functions later; they don't
change their shape.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    func: Callable[..., str]
    description: str
    parameters: dict  # plain JSON schema; each provider adapts it to its own wire format


def bash(command: str) -> str:
    """Run a shell command and return its combined stdout and stderr."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired as expired:
        # Hand the failure back as a result, not an exception - a slow
        # command is the model's problem to work around, not a reason to
        # take the process down.
        return (
            f"Timed out after {expired.timeout}s and was killed. "
            "Narrow it down rather than re-running the same thing."
        )
    return (result.stdout + result.stderr) or "(no output)"


def read_file(path: str) -> str:
    """Read a file and return its contents."""
    try:
        return Path(path).read_text()
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
    """Write content to a file, creating it (and overwriting it) as needed."""
    try:
        Path(path).write_text(content)
    except OSError as error:
        return f"Error writing {path}: {error}"
    return f"Wrote {len(content)} characters to {path}"


def str_replace(path: str, old_str: str, new_str: str) -> str:
    """Replace one exact-match occurrence of old_str with new_str in a file."""
    try:
        text = Path(path).read_text()
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
        Path(path).write_text(text.replace(old_str, new_str, 1))
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
    """Look up a tool by name and run it.

    No permission gate yet (step 6), no error containment yet (step 4's
    inner loop adds that) - this exists only so a smoke test can prove a
    tool call round-trips end to end.
    """
    return TOOLS_BY_NAME[name].func(**args)

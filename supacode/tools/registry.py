"""Flat tool registry.

Every tool the agent can call is declared here once, as a ToolSpec. Only
`bash` is wired up for now, enough to prove tool-calling works end to end
before read_file/write_file/str_replace/write_todos join this same list
(step 5).

No ToolKind yet on purpose: that classification (read_only / file_edit /
other) only earns its keep once the permission gate (step 6) exists to
read it. Adding it now would be a field nothing consumes. It goes back on
ToolSpec in that step, not before.

No sandbox (step 7), no permission check (step 6), no output capping
(step 8) yet - bash runs a raw subprocess. Those wrap this same function
later; they don't change its shape.
"""

import subprocess
from dataclasses import dataclass
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
]

TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def call_tool(name: str, args: dict) -> str:
    """Look up a tool by name and run it.

    No permission gate yet (step 6), no error containment yet (step 4's
    inner loop adds that) - this exists only so a smoke test can prove a
    tool call round-trips end to end.
    """
    return TOOLS_BY_NAME[name].func(**args)

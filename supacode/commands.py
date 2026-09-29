"""Slash commands. Anything typed starting with / lands here.

Empty for now - no /rewind, /sessions, /compact yet, since those depend on
session persistence and compaction, which don't exist until later steps.
This just proves the routing shape: handle(command, messages) -> messages.
"""

COMMANDS = {}


def handle(command, messages):
    print(f"Unknown command: {command}")
    return messages

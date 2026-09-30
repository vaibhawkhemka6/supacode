"""Picks the active provider (config.PROVIDER) once at import time.

The actual client construction + response normalization lives in
models/openai/provider.py and models/anthropic/provider.py. This module
wraps that provider's call() as call_llm, so the rest of the agent
(agent.py) never has to know which provider answered.

call_llm defaults to sending the full TOOLS registry on every call unless
the caller passes its own `tools`. Without this, SYSTEM_PROMPT's "use the
bash tool" is a promise the API request doesn't back up: no tools reach
the model, so it fakes tool-call-shaped text instead of emitting a real
one. Callers that haven't built an execution loop yet (agent.py, for now)
still get real tool definitions on the wire - they just don't do anything
with tool_calls in the response yet. That's fine; it's still correct
behavior, just an unused capability until the inner loop (step 4) reads it.
"""

from . import config
from .tools import TOOLS

SYSTEM_PROMPT = """
You are a coding agent. Your job is to code. Always code.
Use the bash tool to run commands and inspect the filesystem.
Use read_file to read files, write_file to create new files, and
str_replace to make targeted edits to existing files.
"""

if config.PROVIDER == "openai":
    from .models.openai.provider import call as _call, client

elif config.PROVIDER == "anthropic":
    from .models.anthropic.provider import call as _call, client

else:
    raise ValueError(
        f"Unknown PROVIDER: {config.PROVIDER!r} (expected 'openai' or 'anthropic')"
    )


def call_llm(messages, tools=None):
    return _call(messages, tools=tools or TOOLS)


if __name__ == "__main__":
    from .tools import call_tool

    user_input = input("Enter your prompt> ")

    message, usage = call_llm(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        tools=TOOLS,
    )

    print("\nAgent: ", message["content"], "\n")

    if message["tool_calls"]:
        tool_call = message["tool_calls"][0]
        print("Tool: ", tool_call["name"], tool_call["arguments"])
        print(call_tool(tool_call["name"], tool_call["arguments"]), "\n")

    print(usage)

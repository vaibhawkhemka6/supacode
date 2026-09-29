"""Picks the active provider (config.PROVIDER) once at import time.

The actual client construction + response normalization lives in
models/openai/provider.py and models/anthropic/provider.py. This module
just re-exports call_llm so the rest of the agent (agent.py) never has
to know which provider answered.
"""

from . import config
from .tools import TOOLS

SYSTEM_PROMPT = """
You are a coding agent. Your job is to code. Always code.
Use the bash tool to run commands and inspect files.
"""

if config.PROVIDER == "openai":
    from .models.openai.provider import call as call_llm, client

elif config.PROVIDER == "anthropic":
    from .models.anthropic.provider import call as call_llm, client

else:
    raise ValueError(
        f"Unknown PROVIDER: {config.PROVIDER!r} (expected 'openai' or 'anthropic')"
    )


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

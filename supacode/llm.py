"""Picks the active provider (config.PROVIDER) once at import time.

The actual client construction + response normalization lives in
models/openai/provider.py and models/anthropic/provider.py. This module
just re-exports call_llm so the rest of the agent (agent.py) never has
to know which provider answered.
"""

from . import config

SYSTEM_PROMPT = """
You are a coding agent. Your job is to code. Always code.
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
    user_input = input("Enter your prompt> ")

    message, usage = call_llm([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ])

    print("\nAgent: ", message["content"], "\n")
    print(usage)

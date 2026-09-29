"""Anthropic provider.

Builds the client and normalizes the response down to the shared
plain-dict message shape ({"role", "content", "tool_calls"}) that the
rest of the agent works with, regardless of which provider answered.
`tools` (a list of registry.ToolSpec) is converted to Anthropic's
tool-use wire format here - the registry itself stays provider-agnostic.
"""

import anthropic as anthropic_sdk

from ... import config

client = anthropic_sdk.Anthropic(base_url=config.BASE_URL, api_key=config.API_KEY)


def _to_schema(tool):
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


def call(messages, tools=None):
    # Anthropic takes the system prompt as its own top-level param, not as
    # a message in the list — so we peel it off here rather than making the
    # rest of the agent think about it.
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else None
    rest = [m for m in messages if m["role"] != "system"]

    kwargs = {"tools": [_to_schema(t) for t in tools]} if tools else {}

    response = client.messages.create(
        model=config.MODEL,
        system=system,
        messages=rest,
        max_tokens=4096,
        **kwargs,
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    message = {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {"id": block.id, "name": block.name, "arguments": block.input}
            for block in response.content
            if block.type == "tool_use"
        ],
    }

    usage = {
        "prompt_tokens": response.usage.input_tokens,
        "completion_tokens": response.usage.output_tokens,
    }

    return message, usage

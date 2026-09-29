"""OpenAI-compatible provider.

Builds the client and normalizes the response down to the shared
plain-dict message shape ({"role", "content", "tool_calls"}) that the
rest of the agent works with, regardless of which provider answered.
`tools` (a list of registry.ToolSpec) is converted to OpenAI's
function-calling wire format here - the registry itself stays
provider-agnostic.
"""

import json

from openai import OpenAI

from ... import config

client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)


def _to_schema(tool):
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def call(messages, tools=None):
    kwargs = {"tools": [_to_schema(t) for t in tools]} if tools else {}

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=messages,
        **kwargs,
    )

    choice = response.choices[0].message
    message = {
        "role": "assistant",
        "content": choice.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            }
            for tc in (choice.tool_calls or [])
        ],
    }

    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }

    return message, usage

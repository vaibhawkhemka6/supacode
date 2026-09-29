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


def _to_api_message(m):
    # Same problem as the Anthropic side: agent.py feeds the normalized
    # output shape straight back in as input. OpenAI *does* have a
    # message-level "tool_calls" field, but it expects
    # {"id", "type": "function", "function": {"name", "arguments": "<json str>"}}
    # - not our simplified {"id", "name", "arguments": <dict>}. Sending the
    # normalized dict straight through would either be silently wrong
    # (arguments as a dict instead of a JSON string) or rejected, so
    # translate it here.
    if m["role"] == "assistant" and m.get("tool_calls"):
        return {
            "role": "assistant",
            "content": m["content"] or None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                }
                for tc in m["tool_calls"]
            ],
        }

    if m["role"] == "tool":
        # Must keep tool_call_id - it's how the model matches this result
        # back to the specific tool_call it made. Dropping it (the generic
        # fallback below would) makes OpenAI reject the request outright.
        return {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]}

    # Plain user/system/assistant turns: pass through role/content only,
    # dropping any other normalized-shape keys (e.g. an empty
    # "tool_calls": []) that don't belong on the wire.
    return {"role": m["role"], "content": m["content"]}


def call(messages, tools=None):
    kwargs = {"tools": [_to_schema(t) for t in tools]} if tools else {}

    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[_to_api_message(m) for m in messages],
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

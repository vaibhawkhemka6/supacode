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


def call(messages, tools=None, on_text=None):
    kwargs = {"tools": [_to_schema(t) for t in tools]} if tools else {}

    # Unlike Anthropic's SDK, openai's stream=True hands back raw deltas
    # with no built-in accumulator - each chunk carries only the next
    # fragment of text or of a tool call's arguments JSON string, keyed
    # by an `index` that ties fragments from the same tool call together
    # across chunks. We do that accumulation by hand below, then only
    # json.loads() each tool call's arguments once, at the end, when the
    # fragments are guaranteed complete - parsing a half-arrived JSON
    # string mid-stream would just raise. stream_options include_usage
    # asks for the one usage-bearing chunk the API otherwise wouldn't send
    # at all in streaming mode.
    stream = client.chat.completions.create(
        model=config.MODEL,
        messages=[_to_api_message(m) for m in messages],
        stream=True,
        stream_options={"include_usage": True},
        **kwargs,
    )

    text_parts = []
    tool_calls = {}  # index -> {"id", "name", "arguments": "<partial json>"}
    usage = None

    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage

        if not chunk.choices:
            continue  # the final usage-only chunk has no choices

        delta = chunk.choices[0].delta

        if delta.content:
            text_parts.append(delta.content)
            if on_text:
                on_text("".join(text_parts))

        for tc_delta in delta.tool_calls or []:
            slot = tool_calls.setdefault(
                tc_delta.index, {"id": None, "name": None, "arguments": ""}
            )
            if tc_delta.id:
                slot["id"] = tc_delta.id
            if tc_delta.function and tc_delta.function.name:
                slot["name"] = tc_delta.function.name
            if tc_delta.function and tc_delta.function.arguments:
                slot["arguments"] += tc_delta.function.arguments

    message = {
        "role": "assistant",
        "content": "".join(text_parts),
        "tool_calls": [
            {
                "id": tool_calls[i]["id"],
                "name": tool_calls[i]["name"],
                "arguments": json.loads(tool_calls[i]["arguments"] or "{}"),
            }
            for i in sorted(tool_calls)
        ],
    }

    usage = {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }

    return message, usage

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


def _to_api_messages(messages):
    """Convert the whole normalized history to Anthropic's wire format.

    Whole-list, not per-message, because tool results need to merge:
    Anthropic has no "tool" role - a tool result becomes a "user" message
    carrying a tool_result content block - and when one assistant turn
    calls several tools, every one of their results must ride in a
    *single* following user message, not one user message per result (the
    inner loop appends one normalized tool-message per call, so without
    this merge two consecutive tool calls would produce two consecutive
    "user" messages, which the SDK rejects).
    """
    wire = []
    for m in messages:
        if m["role"] == "system":
            continue

        if m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if wire and wire[-1].get("_tool_result_batch"):
                wire[-1]["content"].append(block)
            else:
                wire.append({"role": "user", "content": [block], "_tool_result_batch": True})
            continue

        if m["role"] == "assistant" and m.get("tool_calls"):
            content = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            content.extend(
                {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]}
                for tc in m["tool_calls"]
            )
            wire.append({"role": "assistant", "content": content})
            continue

        # Plain user/assistant turns: pass through role/content only,
        # dropping any other normalized-shape keys (e.g. an empty
        # "tool_calls": []) that Anthropic's strict schema would reject.
        wire.append({"role": m["role"], "content": m["content"]})

    for m in wire:
        m.pop("_tool_result_batch", None)  # internal marker, not part of the wire shape
    return wire


def call(messages, tools=None):
    # Anthropic takes the system prompt as its own top-level param, not as
    # a message in the list — so we peel it off here rather than making the
    # rest of the agent think about it.
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else None
    rest = _to_api_messages(messages)

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

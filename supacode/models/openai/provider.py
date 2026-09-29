"""OpenAI-compatible provider.

Builds the client and normalizes the response down to the shared
plain-dict message shape ({"role": ..., "content": ...}) that the rest
of the agent works with, regardless of which provider answered.
"""

from openai import OpenAI

from ... import config

client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)


def call(messages):
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=messages,
    )

    choice = response.choices[0].message
    message = {"role": "assistant", "content": choice.content or ""}

    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }

    return message, usage

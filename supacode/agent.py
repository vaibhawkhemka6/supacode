from . import commands
from .llm import SYSTEM_PROMPT, call_llm
from .tools import TOOLS, call_tool


def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("> ")
        if not user_input:
            break

        if user_input.startswith("/"):
            messages = commands.handle(user_input, messages)
            continue

        messages.append({"role": "user", "content": user_input})

        while True:  # inner loop: keep calling until the model stops asking for tools
            message, usage = call_llm(messages, tools=TOOLS)
            messages.append(message)

            if message["content"]:
                print("\nAgent:", message["content"], "\n")

            if not message["tool_calls"]:
                break

            for tool_call in message["tool_calls"]:
                result = call_tool(tool_call["name"], tool_call["arguments"])
                print(f"Tool: {tool_call['name']} {tool_call['arguments']}")
                print(result, "\n")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result,
                })


if __name__ == "__main__":
    main()

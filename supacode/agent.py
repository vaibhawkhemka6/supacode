from . import commands
from .llm import SYSTEM_PROMPT, call_llm
from .tools import TOOLS, call_tool
from .ui import ui


def main():
    ui.banner()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = ui.ask()
        if not user_input:
            break

        if user_input.startswith("/"):
            messages = commands.handle(user_input, messages)
            continue

        messages.append({"role": "user", "content": user_input})

        while True:  # inner loop: keep calling until the model stops asking for tools
            with ui.working():
                message, usage = call_llm(messages, tools=TOOLS)
            messages.append(message)
            ui.usage(usage)

            if message["content"]:
                ui.agent(message["content"])

            if not message["tool_calls"]:
                break

            for tool_call in message["tool_calls"]:
                result = call_tool(tool_call["name"], tool_call["arguments"])
                ui.tool(tool_call["name"], tool_call["arguments"], result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result,
                })


if __name__ == "__main__":
    main()

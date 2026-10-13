from . import commands, history, session
from .models.llm import SYSTEM_PROMPT, call_llm
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
            session.save(messages)
            continue

        messages.append({"role": "user", "content": user_input})

        while True:  # inner loop: keep calling until the model stops asking for tools
            with ui.stream() as on_text:
                message, usage = call_llm(messages, tools=TOOLS, on_text=on_text)
            messages.append(message)
            session.save(messages)
            ui.usage(usage)

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
                session.save(messages)

        history.sweep()  # turn's over: bin its temp files, win or not
        history.strip(messages)  # ...and shrink the tool output it produced


if __name__ == "__main__":
    main()

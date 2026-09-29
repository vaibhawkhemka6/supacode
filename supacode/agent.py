from . import commands
from .llm import SYSTEM_PROMPT, call_llm


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

        message, usage = call_llm(messages)
        messages.append(message)

        if message["content"]:
            print("\nAgent:", message["content"], "\n")


if __name__ == "__main__":
    main()

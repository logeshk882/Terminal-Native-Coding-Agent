import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "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"))

MODEL = "gpt-4o"

def main():
    messages = []
    print("Phase 1 agent (OpenAI). Type 'exit' to quit.\n")

    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )

        reply_text = response.choices[0].message.content

        print(f"\nassistant> {reply_text}\n")

        messages.append({"role": "assistant", "content": reply_text})


if __name__ == "__main__":
    main()
"""
Phase 2: One tool - read_file.
Goal: prove the full loop - model decides to call a tool, we execute it,
we feed the result back, model uses it to answer.
"""

import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL = "gpt-4o"

# ---- Tool implementation ----

def read_file(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        # Return the error as a string - let the model see it and react,
        # don't crash the program.
        return f"ERROR reading {path}: {e}"


# ---- Tool schema (what we tell the model is available) ----

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute path to the file to read.",
                    }
                },
                "required": ["path"],
            },
        },
    }
]

# Map tool name -> actual python function, used to dispatch calls
TOOL_FUNCTIONS = {
    "read_file": read_file,
}


def main():
    messages = []
    print("Phase 2 agent. Type 'exit' to quit.\n")

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
            tools=TOOLS,
        )

        reply = response.choices[0].message

        # Case 1: model wants to call a tool
        if reply.tool_calls:
            # append the assistant's tool-call message to history first
            messages.append(reply)

            for tool_call in reply.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                print(f"\n[tool call] {fn_name}({fn_args})")

                fn = TOOL_FUNCTIONS.get(fn_name)
                result = fn(**fn_args) if fn else f"ERROR: unknown tool {fn_name}"

                # feed the tool result back into the conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            # call the model again so it can use the tool result
            followup = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
            )
            followup_reply = followup.choices[0].message
            print(f"\nclaude> {followup_reply.content}\n")
            messages.append(followup_reply)

        # Case 2: model just replied with text, no tool needed
        else:
            print(f"\nclaude> {reply.content}\n")
            messages.append(reply)


if __name__ == "__main__":
    main()
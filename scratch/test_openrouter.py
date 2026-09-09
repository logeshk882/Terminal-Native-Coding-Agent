import asyncio
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("OPENROUTER_API_KEY", "").strip('"').strip("'")
print("Using OpenRouter key:", key[:10] + "...")

models_to_test = [
    "openrouter/auto",
    "openai/gpt-3.5-turbo",
    "openai/gpt-4o-mini",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.1-8b-instruct:free",
]

async def test_models():
    async with httpx.AsyncClient() as client:
        for model in models_to_test:
            try:
                res = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/logeshk882/Terminal-Native-Coding-Agent",
                        "X-Title": "TMCA",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "Hello! What is recursion in 1 sentence?"}],
                    },
                    timeout=10.0,
                )
                print(f"Model: {model:<40} -> Status: {res.status_code}")
                if res.status_code == 200:
                    print("  Response:", res.json()["choices"][0]["message"]["content"])
                    break
                else:
                    print("  Error:", res.text[:150])
            except Exception as e:
                print(f"Model: {model:<40} -> Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_models())

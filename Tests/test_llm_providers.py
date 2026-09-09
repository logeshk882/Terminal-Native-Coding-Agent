"""
Unit tests for OpenAI and OpenRouter provider implementations.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.LLM.base import FinishReason, LLMConfig, Message, Role, ToolDefinition
from src.LLM.openAI import OpenAIProvider
from src.LLM.openrouter import OpenRouterProvider


class TestLLMProviders(unittest.TestCase):

    def test_openai_payload_preparation(self):
        config = LLMConfig(
            model_name="gpt-4o",
            temperature=0.2,
            system_instruction="System prompt test",
            api_key="sk-test-key",
        )
        provider = OpenAIProvider(config=config)

        messages = [Message.user("Hello")]
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]

        payload = provider._prepare_payload(messages, tools=tools, config=config, stream=False)

        self.assertEqual(payload["model"], "gpt-4o")
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["content"], "Hello")
        self.assertIn("tools", payload)
        self.assertEqual(payload["tools"][0]["function"]["name"], "test_tool")

    def test_openai_generate_mock(self):
        async def run_test():
            config = LLMConfig(model_name="gpt-4o", api_key="sk-test-key")

            mock_response_data = {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello user!",
                            "tool_calls": [
                                {
                                    "id": "call_99",
                                    "type": "function",
                                    "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
            }

            mock_http_response = MagicMock(spec=httpx.Response)
            mock_http_response.status_code = 200
            mock_http_response.json.return_value = mock_response_data
            mock_http_response.raise_for_status = MagicMock()

            mock_client = MagicMock(spec=httpx.AsyncClient)
            mock_client.post = AsyncMock(return_value=mock_http_response)

            provider = OpenAIProvider(config=config, client=mock_client)
            res = await provider.generate([Message.user("Hi")])

            self.assertEqual(res.message.role, Role.ASSISTANT)
            self.assertEqual(res.message.content, "Hello user!")
            self.assertEqual(len(res.message.tool_calls), 1)
            self.assertEqual(res.message.tool_calls[0].name, "read_file")
            self.assertEqual(res.finish_reason, FinishReason.TOOL_CALLS)
            self.assertEqual(res.usage.total_tokens, 23)

        asyncio.run(run_test())

    def test_openrouter_headers_and_config(self):
        config = LLMConfig(model_name="anthropic/claude-3.5-sonnet", api_key="sk-or-key")
        provider = OpenRouterProvider(config=config)

        self.assertEqual(provider.provider_name, "openrouter")
        self.assertTrue(provider._get_base_url().startswith("https://openrouter.ai"))
        headers = provider._get_headers(config)
        self.assertEqual(headers["Authorization"], "Bearer sk-or-key")
        self.assertIn("HTTP-Referer", headers)
        self.assertIn("X-Title", headers)


if __name__ == "__main__":
    unittest.main()

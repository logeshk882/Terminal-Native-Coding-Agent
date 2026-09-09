"""
Unit tests for src/LLM/base.py
"""

import asyncio
import unittest
from typing import AsyncIterator, List, Optional

from src.LLM.base import (
    BaseLLMProvider,
    FinishReason,
    LLMChunk,
    LLMConfig,
    LLMResponse,
    Message,
    Role,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)


class MockLLMProvider(BaseLLMProvider):
    """Concrete mock implementation of BaseLLMProvider for testing."""

    @property
    def provider_name(self) -> str:
        return "mock_provider"

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        reply_content = f"Echo: {messages[-1].content}" if messages and isinstance(messages[-1].content, str) else "Echo"
        return LLMResponse(
            message=Message.assistant(content=reply_content),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model=config.model_name if config else "mock-model",
        )

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[LLMChunk]:
        tokens = ["Hello", " world", "!"]
        for token in tokens:
            yield LLMChunk(delta_content=token)
        yield LLMChunk(
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
        )


class TestLLMBaseContract(unittest.TestCase):

    def test_message_creation_and_dict_conversion(self):
        sys_msg = Message.system("You are a helpful assistant.")
        self.assertEqual(sys_msg.role, Role.SYSTEM)
        self.assertEqual(sys_msg.to_dict(), {"role": "system", "content": "You are a helpful assistant."})

        user_msg = Message.user("Hello AI!")
        self.assertEqual(user_msg.role, Role.USER)
        self.assertEqual(user_msg.to_dict(), {"role": "user", "content": "Hello AI!"})

        tool_call = ToolCall(id="call_123", name="read_file", arguments={"path": "main.py"})
        self.assertEqual(tool_call.get_arguments_dict(), {"path": "main.py"})

        asst_msg = Message.assistant(tool_calls=[tool_call])
        self.assertEqual(asst_msg.role, Role.ASSISTANT)
        asst_dict = asst_msg.to_dict()
        self.assertIn("tool_calls", asst_dict)
        self.assertEqual(asst_dict["tool_calls"][0]["function"]["name"], "read_file")

        tool_result = Message.tool(content="file contents", tool_call_id="call_123", name="read_file")
        self.assertEqual(tool_result.to_dict(), {
            "role": "tool",
            "content": "file contents",
            "name": "read_file",
            "tool_call_id": "call_123",
        })

    def test_tool_definition(self):
        tool_def = ToolDefinition(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        as_dict = tool_def.to_dict()
        self.assertEqual(as_dict["type"], "function")
        self.assertEqual(as_dict["function"]["name"], "get_weather")

    def test_token_usage_addition(self):
        u1 = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        u2 = TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        u3 = u1 + u2
        self.assertEqual(u3.prompt_tokens, 30)
        self.assertEqual(u3.completion_tokens, 15)
        self.assertEqual(u3.total_tokens, 45)

    def test_mock_provider_generate(self):
        async def run_test():
            config = LLMConfig(model_name="mock-v1")
            provider = MockLLMProvider(config=config)
            self.assertEqual(provider.provider_name, "mock_provider")
            self.assertTrue(provider.supports_tools())
            self.assertTrue(provider.supports_streaming())

            messages = [Message.user("Hello World")]
            response = await provider.generate(messages, config=config)

            self.assertEqual(response.message.content, "Echo: Hello World")
            self.assertEqual(response.finish_reason, FinishReason.STOP)
            self.assertEqual(response.usage.total_tokens, 15)
            self.assertEqual(response.model, "mock-v1")

        asyncio.run(run_test())

    def test_mock_provider_generate_stream(self):
        async def run_test():
            provider = MockLLMProvider()
            chunks = []
            final_usage = None
            async for chunk in provider.generate_stream([Message.user("Test")]):
                if chunk.delta_content:
                    chunks.append(chunk.delta_content)
                if chunk.usage:
                    final_usage = chunk.usage

            self.assertEqual("".join(chunks), "Hello world!")
            self.assertIsNotNone(final_usage)
            self.assertEqual(final_usage.total_tokens, 13)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()

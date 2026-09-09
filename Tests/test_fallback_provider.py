"""
Unit tests for FallbackLLMProvider and OmniRouteProvider.
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
    TokenUsage,
    ToolDefinition,
)
from src.LLM.fallback import FallbackLLMProvider
from src.LLM.omniroute import OmniRouteProvider


class FailingProvider(BaseLLMProvider):

    @property
    def provider_name(self) -> str:
        return "failing_provider"

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        raise RuntimeError("401 Unauthorized simulated failure")

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[LLMChunk]:
        raise RuntimeError("Streaming failure")


class SuccessfulProvider(BaseLLMProvider):

    @property
    def provider_name(self) -> str:
        return "successful_provider"

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        return LLMResponse(
            message=Message.assistant(content="Fallback success response"),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            model="fallback-model",
        )

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[LLMChunk]:
        yield LLMChunk(delta_content="Fallback stream")


class TestFallbackProvider(unittest.TestCase):

    def test_fallback_failover_sequence(self):
        async def run_test():
            p1 = FailingProvider()
            p2 = SuccessfulProvider()
            fallback_provider = FallbackLLMProvider(providers=[p1, p2])

            self.assertIn("fallback", fallback_provider.provider_name)

            response = await fallback_provider.generate([Message.user("Hello")])
            self.assertEqual(response.message.content, "Fallback success response")

        asyncio.run(run_test())

    def test_omniroute_provider_name_and_url(self):
        config = LLMConfig(model_name="omni-v1", api_key="sk-omni-test")
        provider = OmniRouteProvider(config=config)
        self.assertEqual(provider.provider_name, "omniroute")
        self.assertTrue(provider._get_base_url().startswith("https://api.omniroute.ai"))


if __name__ == "__main__":
    unittest.main()

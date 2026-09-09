"""
Fallback LLM Provider allowing automatic multi-provider failover.
"""

import logging
from typing import AsyncIterator, List, Optional

from src.LLM.base import (
    BaseLLMProvider,
    LLMChunk,
    LLMConfig,
    LLMResponse,
    Message,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class FallbackLLMProvider(BaseLLMProvider):
    """
    Wraps multiple LLM providers and automatically falls back to the next
    available provider if the primary provider fails (e.g. 401, 429, 500 error).
    """

    def __init__(self, providers: List[BaseLLMProvider], config: Optional[LLMConfig] = None) -> None:
        super().__init__(config=config)
        if not providers:
            raise ValueError("FallbackLLMProvider requires at least one provider.")
        self.providers = providers

    @property
    def provider_name(self) -> str:
        names = [p.provider_name for p in self.providers]
        return f"fallback({', '.join(names)})"

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        last_exception = None
        for provider in self.providers:
            try:
                logger.info("Attempting generation with provider '%s'...", provider.provider_name)
                return await provider.generate(messages, tools=tools, config=config)
            except Exception as e:
                logger.warning(
                    "Provider '%s' failed with error: %s. Falling back to next provider...",
                    provider.provider_name,
                    e,
                )
                last_exception = e

        if last_exception:
            raise last_exception
        raise RuntimeError("All providers in fallback chain failed.")

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[LLMChunk]:
        last_exception = None
        for provider in self.providers:
            yielded_any = False
            try:
                async for chunk in provider.generate_stream(messages, tools=tools, config=config):
                    yielded_any = True
                    yield chunk
                return
            except Exception as e:
                if yielded_any:
                    # Chunks were already streamed to client, cannot cleanly switch provider mid-stream
                    raise e
                logger.warning(
                    "Provider '%s' streaming failed with error: %s. Falling back to next provider...",
                    provider.provider_name,
                    e,
                )
                last_exception = e

        if last_exception:
            raise last_exception
        raise RuntimeError("All providers in fallback chain failed.")

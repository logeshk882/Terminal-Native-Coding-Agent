"""
OpenRouter LLM Provider implementation with auto fallback models on 402/404 errors.
"""

import logging
import os
from typing import AsyncIterator, Dict, List, Optional
import httpx

from src.LLM.base import LLMChunk, LLMConfig, LLMResponse, Message, ToolDefinition
from src.LLM.openAI import OpenAIProvider

logger = logging.getLogger(__name__)

# Fallback models available on OpenRouter when primary model fails
FALLBACK_MODELS = [
    "openrouter/auto",
    "openai/gpt-3.5-turbo",
    "openai/gpt-4o-mini",
]


class OpenRouterProvider(OpenAIProvider):
    """
    OpenRouter LLM Provider allowing access to 200+ models via OpenAI-compatible API,
    with automatic fallback to openrouter/auto if 402/404 error occurs.
    """

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def _get_api_key(self, config: Optional[LLMConfig] = None) -> str:
        if config and config.api_key:
            return config.api_key
        if self.config and self.config.api_key:
            return self.config.api_key
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY environment variable or config.api_key is required")
        return key.strip().strip('"').strip("'")

    def _get_base_url(self, config: Optional[LLMConfig] = None) -> str:
        if config and config.api_base:
            return config.api_base.rstrip("/")
        if self.config and self.config.api_base:
            return self.config.api_base.rstrip("/")
        url = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        return url.rstrip("/")

    def _get_headers(self, config: Optional[LLMConfig] = None) -> Dict[str, str]:
        headers = super()._get_headers(config)
        headers["HTTP-Referer"] = os.getenv("TMCA_APP_URL", "https://github.com/logeshk882/Terminal-Native-Coding-Agent")
        headers["X-Title"] = "TMCA Terminal-Native Coding Assistant"
        return headers

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        try:
            return await super().generate(messages, tools=tools, config=config)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (402, 404):
                logger.warning(
                    "OpenRouter returned status %d for requested model. Retrying with fallback model 'openrouter/auto'...",
                    e.response.status_code,
                )
                cfg = config or self.config
                fallback_model = os.getenv("OPENROUTER_FALLBACK_MODEL", FALLBACK_MODELS[0])
                new_cfg = LLMConfig(
                    model_name=fallback_model,
                    temperature=cfg.temperature if cfg else 0.7,
                    top_p=cfg.top_p if cfg else 1.0,
                    max_tokens=cfg.max_tokens if cfg else None,
                    api_key=self._get_api_key(cfg),
                    api_base=self._get_base_url(cfg),
                )
                return await super().generate(messages, tools=tools, config=new_cfg)
            raise e

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[LLMChunk]:
        try:
            async for chunk in super().generate_stream(messages, tools=tools, config=config):
                yield chunk
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (402, 404):
                logger.warning(
                    "OpenRouter returned status %d. Retrying stream with fallback model 'openrouter/auto'...",
                    e.response.status_code,
                )
                cfg = config or self.config
                fallback_model = os.getenv("OPENROUTER_FALLBACK_MODEL", FALLBACK_MODELS[0])
                new_cfg = LLMConfig(
                    model_name=fallback_model,
                    temperature=cfg.temperature if cfg else 0.7,
                    top_p=cfg.top_p if cfg else 1.0,
                    max_tokens=cfg.max_tokens if cfg else None,
                    api_key=self._get_api_key(cfg),
                    api_base=self._get_base_url(cfg),
                )
                async for chunk in super().generate_stream(messages, tools=tools, config=new_cfg):
                    yield chunk
            else:
                raise e

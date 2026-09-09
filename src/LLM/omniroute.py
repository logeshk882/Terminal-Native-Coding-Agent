"""
OmniRoute LLM Provider implementation.
"""

import os
from typing import Dict, Optional

from src.LLM.base import LLMConfig
from src.LLM.openAI import OpenAIProvider


class OmniRouteProvider(OpenAIProvider):
    """
    OmniRoute LLM Provider for routing requests across multiple LLM endpoints.
    """

    @property
    def provider_name(self) -> str:
        return "omniroute"

    def _get_api_key(self, config: Optional[LLMConfig] = None) -> str:
        if config and config.api_key:
            return config.api_key
        if self.config and self.config.api_key:
            return self.config.api_key
        key = os.getenv("OMNIROUTE_API_KEY")
        if not key:
            key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OMNIROUTE_API_KEY environment variable or config.api_key is required")
        return key.strip().strip('"').strip("'")

    def _get_base_url(self, config: Optional[LLMConfig] = None) -> str:
        if config and config.api_base:
            return config.api_base.rstrip("/")
        if self.config and self.config.api_base:
            return self.config.api_base.rstrip("/")
        url = os.getenv("OMNIROUTE_API_BASE", "https://api.omniroute.ai/v1")
        return url.rstrip("/")

    def _get_headers(self, config: Optional[LLMConfig] = None) -> Dict[str, str]:
        headers = super()._get_headers(config)
        headers["X-Title"] = "TMCA Terminal-Native Coding Assistant"
        return headers

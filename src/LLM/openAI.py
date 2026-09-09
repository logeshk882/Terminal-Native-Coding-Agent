"""
OpenAI LLM Provider implementation using HTTPX.
"""

import json
import os
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx

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


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider communicating over raw HTTP via httpx.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        super().__init__(config=config)
        self._client = client

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_api_key(self, config: Optional[LLMConfig] = None) -> str:
        if config and config.api_key:
            return config.api_key
        if self.config and self.config.api_key:
            return self.config.api_key
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                key = os.getenv("OPENAI_API_KEY")
            except ImportError:
                pass
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable or config.api_key is required")
        return key.strip().strip('"').strip("'")

    def _get_base_url(self, config: Optional[LLMConfig] = None) -> str:
        if config and config.api_base:
            return config.api_base.rstrip("/")
        if self.config and self.config.api_base:
            return self.config.api_base.rstrip("/")
        url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        return url.rstrip("/")

    def _prepare_payload(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        cfg = config or self.config
        model = cfg.model_name if cfg and cfg.model_name else "gpt-4o"

        formatted_messages = [msg.to_dict() for msg in messages]

        # Prepend system instruction if set in config and not present in messages
        if cfg and cfg.system_instruction and not any(m.role == Role.SYSTEM for m in messages):
            formatted_messages.insert(0, {"role": "system", "content": cfg.system_instruction})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "stream": stream,
        }

        if cfg:
            payload["temperature"] = cfg.temperature
            payload["top_p"] = cfg.top_p
            if cfg.max_tokens is not None:
                payload["max_tokens"] = cfg.max_tokens
            if cfg.stop_sequences:
                payload["stop"] = cfg.stop_sequences
            if cfg.extra_kwargs:
                payload.update(cfg.extra_kwargs)

        if tools:
            payload["tools"] = [tool.to_dict() for tool in tools]
            payload["tool_choice"] = "auto"

        return payload

    def _get_headers(self, config: Optional[LLMConfig] = None) -> Dict[str, str]:
        api_key = self._get_api_key(config)
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        url = f"{self._get_base_url(config)}/chat/completions"
        headers = self._get_headers(config)
        payload = self._prepare_payload(messages, tools, config, stream=False)
        timeout = config.timeout if config else (self.config.timeout if self.config else 60.0)

        should_close = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=timeout)
            should_close = True

        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            msg_data = choice["message"]

            # Parse tool calls if present
            tool_calls = None
            if "tool_calls" in msg_data and msg_data["tool_calls"]:
                tool_calls = []
                for tc in msg_data["tool_calls"]:
                    tool_calls.append(
                        ToolCall(
                            id=tc["id"],
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        )
                    )

            msg = Message(
                role=Role.ASSISTANT,
                content=msg_data.get("content"),
                tool_calls=tool_calls,
            )

            raw_finish = choice.get("finish_reason", "stop")
            try:
                finish_reason = FinishReason(raw_finish)
            except ValueError:
                finish_reason = FinishReason.STOP

            usage_data = data.get("usage", {})
            usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

            return LLMResponse(
                message=msg,
                finish_reason=finish_reason,
                usage=usage,
                model=data.get("model", payload["model"]),
                raw_response=data,
            )
        finally:
            if should_close:
                await client.aclose()

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[LLMChunk]:
        url = f"{self._get_base_url(config)}/chat/completions"
        headers = self._get_headers(config)
        payload = self._prepare_payload(messages, tools, config, stream=True)
        timeout = config.timeout if config else (self.config.timeout if self.config else 60.0)

        should_close = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=timeout)
            should_close = True

        try:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk_json = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk_json.get("choices", [])
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})

                    content = delta.get("content")
                    delta_tools = delta.get("tool_calls")
                    finish_reason = None
                    if choice.get("finish_reason"):
                        try:
                            finish_reason = FinishReason(choice["finish_reason"])
                        except ValueError:
                            finish_reason = FinishReason.STOP

                    usage = None
                    if "usage" in chunk_json and chunk_json["usage"]:
                        u = chunk_json["usage"]
                        usage = TokenUsage(
                            prompt_tokens=u.get("prompt_tokens", 0),
                            completion_tokens=u.get("completion_tokens", 0),
                            total_tokens=u.get("total_tokens", 0),
                        )

                    yield LLMChunk(
                        delta_content=content,
                        delta_tool_calls=delta_tools,
                        finish_reason=finish_reason,
                        usage=usage,
                        raw_chunk=chunk_json,
                    )
        finally:
            if should_close:
                await client.aclose()

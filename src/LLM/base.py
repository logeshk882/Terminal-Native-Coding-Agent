from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union


class Role(str, Enum):
    """Message roles for conversation context."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Reason why the model stopped generating output."""
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


@dataclass
class ToolCall:
    """Represents a tool/function call invoked by the assistant."""
    id: str
    name: str
    arguments: Union[Dict[str, Any], str]

    def get_arguments_dict(self) -> Dict[str, Any]:
        """Returns arguments as a dictionary, parsing JSON string if needed."""
        if isinstance(self.arguments, dict):
            return self.arguments
        if isinstance(self.arguments, str):
            try:
                return json.loads(self.arguments)
            except json.JSONDecodeError:
                return {}
        return {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert ToolCall to standard dictionary format."""
        args = self.arguments if isinstance(self.arguments, str) else json.dumps(self.arguments)
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": args
            }
        }


@dataclass
class Message:
    """Represents a message in the conversation history."""
    role: Union[Role, str]
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.role, str):
            try:
                self.role = Role(self.role)
            except ValueError:
                pass

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: Union[str, List[Dict[str, Any]]]) -> "Message":
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls,
        content: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None
    ) -> "Message":
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: Optional[str] = None) -> "Message":
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Message to API payload dictionary."""
        role_str = self.role.value if isinstance(self.role, Role) else str(self.role)
        res: Dict[str, Any] = {"role": role_str}
        if self.content is not None:
            res["content"] = self.content
        if self.name is not None:
            res["name"] = self.name
        if self.tool_call_id is not None:
            res["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            res["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return res


@dataclass
class ToolDefinition:
    """Schema definition for a tool exposed to an LLM."""
    name: str
    description: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible function tool specification."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


@dataclass
class TokenUsage:
    """Tracks token consumption for an execution request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
        )

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class LLMConfig:
    """Configuration options for an LLM call."""
    model_name: str
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)
    system_instruction: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: float = 60.0
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Complete response returned by an LLM provider."""
    message: Message
    finish_reason: FinishReason = FinishReason.STOP
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    raw_response: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message.to_dict(),
            "finish_reason": self.finish_reason.value,
            "usage": self.usage.to_dict(),
            "model": self.model,
        }


@dataclass
class LLMChunk:
    """Streaming chunk delta yielded during generation."""
    delta_content: Optional[str] = None
    delta_tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[FinishReason] = None
    usage: Optional[TokenUsage] = None
    raw_chunk: Optional[Any] = None


class BaseLLMProvider(ABC):
    """Abstract Base Class for all LLM providers in TMCA."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique identifier for the provider (e.g. 'openai', 'openrouter', 'omniroute')."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate a non-streaming completion from the LLM provider.
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[LLMChunk]:
        """
        Stream completion chunks dynamically from the LLM provider.
        """
        pass

    def supports_tools(self) -> bool:
        """Returns True if provider supports function calling."""
        return True

    def supports_streaming(self) -> bool:
        """Returns True if provider supports streaming responses."""
        return True

    def supports_vision(self) -> bool:
        """Returns True if provider supports multi-modal image inputs."""
        return False

    async def count_tokens(self, messages: List[Message]) -> int:
        """
        Estimate token count for input messages.
        Default naive implementation based on word count; providers should override.
        """
        total_words = 0
        for msg in messages:
            if isinstance(msg.content, str):
                total_words += len(msg.content.split())
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and "text" in part:
                        total_words += len(part["text"].split())
        return int(total_words * 1.33)

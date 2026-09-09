"""
Base abstractions for tools, arguments validation, and tool execution observations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, ValidationError

from src.LLM.base import ToolDefinition


@dataclass
class ToolObservation:
    """Structured result returned from a tool execution."""
    tool_name: str
    tool_call_id: str
    success: bool
    output: str
    exit_code: Optional[int] = 0
    duration_ms: float = 0.0
    error: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None

    def format_for_llm(self) -> str:
        """Format observation as a concise string for LLM tool result messages."""
        if self.success:
            return self.output
        error_msg = self.error or "Unknown tool execution failure"
        if self.output:
            return f"Error: {error_msg}\nOutput:\n{self.output}"
        return f"Error: {error_msg}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "success": self.success,
            "output": self.output,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "raw_data": self.raw_data,
        }


class Tool(ABC):
    """Abstract Base Class for all executable tools in TMCA."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human and LLM-readable description of what the tool does."""
        pass

    @property
    @abstractmethod
    def args_model(self) -> Type[BaseModel]:
        """Pydantic model class defining the tool's expected input schema."""
        pass

    def get_definition(self) -> ToolDefinition:
        """Generate a ToolDefinition matching OpenAI JSON Schema tool specs."""
        schema = self.args_model.model_json_schema()

        # Extract parameters dict matching standard function calling format
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        parameters = {
            "type": "object",
            "properties": properties,
        }
        if required:
            parameters["required"] = required

        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=parameters,
        )

    def validate_args(self, kwargs: Dict[str, Any]) -> BaseModel:
        """Validate keyword arguments against the Pydantic input model."""
        return self.args_model(**kwargs)

    @abstractmethod
    async def run(self, args: BaseModel) -> ToolObservation:
        """Execute the core logic of the tool with validated Pydantic args."""
        pass

    async def execute(self, tool_call_id: str, kwargs: Dict[str, Any]) -> ToolObservation:
        """
        Validate input arguments and execute tool, capturing errors and execution timing.
        """
        start_time = time.perf_counter()
        try:
            validated_args = self.validate_args(kwargs)
            observation = await self.run(validated_args)
            observation.tool_call_id = tool_call_id
            observation.tool_name = self.name
            observation.duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return observation
        except ValidationError as ve:
            duration = round((time.perf_counter() - start_time) * 1000, 2)
            return ToolObservation(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                output="",
                exit_code=1,
                duration_ms=duration,
                error=f"Invalid arguments for tool '{self.name}': {ve}",
            )
        except Exception as e:
            duration = round((time.perf_counter() - start_time) * 1000, 2)
            return ToolObservation(
                tool_name=self.name,
                tool_call_id=tool_call_id,
                success=False,
                output="",
                exit_code=1,
                duration_ms=duration,
                error=f"Execution error in tool '{self.name}': {str(e)}",
            )

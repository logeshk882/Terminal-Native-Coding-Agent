"""
Executor for validating and running tool calls requested by the agent.
"""

from typing import Dict, Any, Optional
from src.LLM.base import ToolCall
from src.tools.base import ToolObservation
from src.tools.registry import ToolRegistry


class ToolExecutor:
    """Dispatches tool calls requested by the LLM to registered Tool implementations."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def execute_tool_call(self, tool_call: ToolCall) -> ToolObservation:
        """
        Execute a ToolCall instance and return a structured ToolObservation.
        """
        tool = self.registry.get(tool_call.name)
        if not tool:
            return ToolObservation(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                success=False,
                output="",
                exit_code=1,
                error=f"Unknown tool '{tool_call.name}'. Registered tools: {list(self.registry._tools.keys())}",
            )

        kwargs = tool_call.get_arguments_dict()
        return await tool.execute(tool_call_id=tool_call.id, kwargs=kwargs)

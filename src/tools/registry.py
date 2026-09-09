"""
Registry for registering, retrieving, and inspecting available tools.
"""

from typing import Dict, List, Optional
from src.LLM.base import ToolDefinition
from src.tools.base import Tool


class ToolRegistry:
    """Central registry of executable agent tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance in the registry."""
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Retrieve a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """List all registered tool instances."""
        return list(self._tools.values())

    def get_tool_definitions(self) -> List[ToolDefinition]:
        """Get tool definitions formatted for LLM prompts."""
        return [tool.get_definition() for tool in self._tools.values()]

"""
Unit tests for Tool infrastructure.
"""

import asyncio
import unittest
from pydantic import BaseModel, Field

from src.LLM.base import ToolCall
from src.tools.base import Tool, ToolObservation
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry


class MultiplyArgs(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


class MultiplyTool(Tool):

    @property
    def name(self) -> str:
        return "multiply"

    @property
    def description(self) -> str:
        return "Multiply two numbers"

    @property
    def args_model(self):
        return MultiplyArgs

    async def run(self, args: MultiplyArgs) -> ToolObservation:
        result = args.a * args.b
        return ToolObservation(
            tool_name=self.name,
            tool_call_id="",
            success=True,
            output=str(result),
        )


class TestToolSubsystem(unittest.TestCase):

    def test_tool_definition_generation(self):
        tool = MultiplyTool()
        definition = tool.get_definition()

        self.assertEqual(definition.name, "multiply")
        self.assertEqual(definition.description, "Multiply two numbers")
        self.assertIn("a", definition.parameters["properties"])
        self.assertIn("b", definition.parameters["properties"])

    def test_registry_and_executor(self):
        async def run_test():
            registry = ToolRegistry()
            tool = MultiplyTool()
            registry.register(tool)

            self.assertEqual(len(registry.list_tools()), 1)
            self.assertEqual(len(registry.get_tool_definitions()), 1)

            executor = ToolExecutor(registry)

            # Test valid execution
            call = ToolCall(id="call_1", name="multiply", arguments={"a": 6, "b": 7})
            obs = await executor.execute_tool_call(call)
            self.assertTrue(obs.success)
            self.assertEqual(obs.output, "42.0")
            self.assertEqual(obs.tool_call_id, "call_1")

            # Test invalid argument execution
            invalid_call = ToolCall(id="call_2", name="multiply", arguments={"a": "not_a_number"})
            invalid_obs = await executor.execute_tool_call(invalid_call)
            self.assertFalse(invalid_obs.success)
            self.assertIn("Invalid arguments", invalid_obs.error)

            # Test unknown tool execution
            unknown_call = ToolCall(id="call_3", name="unknown_tool", arguments={})
            unknown_obs = await executor.execute_tool_call(unknown_call)
            self.assertFalse(unknown_obs.success)
            self.assertIn("Unknown tool", unknown_obs.error)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()

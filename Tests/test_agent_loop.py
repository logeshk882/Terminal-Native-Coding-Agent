"""
Integration tests for TMCAAgent and AgentLoop.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import AsyncIterator, List, Optional

from src.Agent.agent import TMCAAgent
from src.Agent.state import AgentStatus
from src.LLM.base import (
    BaseLLMProvider,
    FinishReason,
    LLMChunk,
    LLMConfig,
    LLMResponse,
    Message,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)


class ScriptedMockProvider(BaseLLMProvider):
    """Mock provider yielding pre-scripted responses for loop testing."""

    def __init__(self, responses: List[LLMResponse]) -> None:
        super().__init__()
        self.responses = responses
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "scripted_mock"

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[LLMChunk]:
        yield LLMChunk(delta_content="Stream output")


class TestAgentLoopIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_end_to_end_agent_tool_loop(self):
        async def run_test():
            # Step 1 response: Request write_file tool
            resp1 = LLMResponse(
                message=Message.assistant(
                    tool_calls=[
                        ToolCall(id="tc1", name="write_file", arguments={"path": "notes.txt", "content": "TMCA test"})
                    ]
                ),
                finish_reason=FinishReason.TOOL_CALLS,
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            )

            # Step 2 response: Final answer
            resp2 = LLMResponse(
                message=Message.assistant(content="Successfully wrote notes.txt file."),
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(prompt_tokens=30, completion_tokens=5, total_tokens=35),
            )

            provider = ScriptedMockProvider([resp1, resp2])
            agent = TMCAAgent(provider=provider, workspace_dir=self.workspace)

            state = await agent.run("Create notes.txt file")

            self.assertEqual(state.status, AgentStatus.FINISHED)
            self.assertEqual(state.iteration, 1)
            self.assertEqual(state.total_usage.total_tokens, 65)
            self.assertEqual(state.last_response_text, "Successfully wrote notes.txt file.")

            # Verify physical file was written to disk by tool execution
            created_file = self.workspace / "notes.txt"
            self.assertTrue(created_file.exists())
            self.assertEqual(created_file.read_text(), "TMCA test")

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()

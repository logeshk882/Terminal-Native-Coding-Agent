"""
Core agent loop implementing the Plan-Act-Observe-Update execution cycle.
"""

import logging
from typing import Callable, Optional

from src.Agent.state import AgentState, AgentStatus
from src.LLM.base import BaseLLMProvider, FinishReason, LLMConfig
from src.tools.base import ToolObservation
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentLoop:
    """Orchestrates interaction between LLM provider, tool executor, and agent state."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        executor: ToolExecutor,
        registry: ToolRegistry,
        state: AgentState,
        config: Optional[LLMConfig] = None,
        on_step: Optional[Callable[[AgentState, Optional[ToolObservation]], None]] = None,
    ) -> None:
        self.provider = provider
        self.executor = executor
        self.registry = registry
        self.state = state
        self.config = config
        self.on_step = on_step

    async def run_task(self, task: str) -> AgentState:
        """
        Execute an agent task through iterative LLM generation and tool call execution.
        """
        self.state.current_task = task
        self.state.status = AgentStatus.RUNNING
        self.state.add_user_message(task)

        while self.state.status == AgentStatus.RUNNING:
            if self.state.iteration >= self.state.max_iterations:
                self.state.status = AgentStatus.FAILED
                logger.warning("Agent loop reached maximum allowed iterations (%d)", self.state.max_iterations)
                break

            tool_defs = self.registry.get_tool_definitions()
            response = await self.provider.generate(
                messages=self.state.messages,
                tools=tool_defs if tool_defs else None,
                config=self.config,
            )

            # Accumulate token metrics
            if response.usage:
                self.state.total_usage = self.state.total_usage + response.usage

            self.state.add_assistant_message(response.message)

            # Stop if model gave final text response without requesting tool calls
            if not response.message.tool_calls or response.finish_reason == FinishReason.STOP and not response.message.tool_calls:
                self.state.status = AgentStatus.FINISHED
                if self.on_step:
                    self.on_step(self.state, None)
                break

            # Execute tool calls
            for tool_call in response.message.tool_calls:
                obs = await self.executor.execute_tool_call(tool_call)
                self.state.add_observation(obs)

                if self.on_step:
                    self.on_step(self.state, obs)

            self.state.iteration += 1

        return self.state

"""
Main TMCAAgent controller unifying provider, tool registry, security policy, and execution loop.
"""

from pathlib import Path
from typing import Callable, Optional, Union

from src.Agent.loop import AgentLoop
from src.Agent.prompt import SystemPromptBuilder
from src.Agent.state import AgentState
from src.LLM.base import BaseLLMProvider, LLMConfig, Message
from src.security.permissions import PermissionManager
from src.tools.builtin.filesystem import (
    ApplyPatchTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from src.tools.builtin.search import SearchFilesTool
from src.tools.builtin.shell import RunCommandTool
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry


class TMCAAgent:
    """
    Terminal-Native Coding Assistant Agent.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        workspace_dir: Union[str, Path] = ".",
        config: Optional[LLMConfig] = None,
        allow_high_risk_commands: bool = False,
    ) -> None:
        self.provider = provider
        self.workspace_dir = Path(workspace_dir).resolve()
        self.config = config

        self.permission_manager = PermissionManager(self.workspace_dir)
        self.registry = ToolRegistry()

        # Register standard built-in tools
        self.registry.register(ListDirectoryTool(self.permission_manager))
        self.registry.register(ReadFileTool(self.permission_manager))
        self.registry.register(WriteFileTool(self.permission_manager))
        self.registry.register(ApplyPatchTool(self.permission_manager))
        self.registry.register(SearchFilesTool(self.permission_manager))
        self.registry.register(RunCommandTool(self.permission_manager, allow_high_risk=allow_high_risk_commands))

        self.executor = ToolExecutor(self.registry)

    async def run(
        self,
        task: str,
        on_step: Optional[Callable] = None,
        max_iterations: int = 20,
    ) -> AgentState:
        """
        Initialize agent state and run an end-to-end task loop.
        """
        system_prompt = SystemPromptBuilder.build_system_prompt(self.workspace_dir)
        state = AgentState(
            messages=[Message.system(system_prompt)],
            max_iterations=max_iterations,
        )

        loop = AgentLoop(
            provider=self.provider,
            executor=self.executor,
            registry=self.registry,
            state=state,
            config=self.config,
            on_step=on_step,
        )

        return await loop.run_task(task)

    def create_state(self, max_iterations: int = 20) -> AgentState:
        """Create a new AgentState initialized with system prompt instructions."""
        system_prompt = SystemPromptBuilder.build_system_prompt(self.workspace_dir)
        return AgentState(
            messages=[Message.system(system_prompt)],
            max_iterations=max_iterations,
        )

    async def run_turn(
        self,
        state: AgentState,
        task: str,
        on_step: Optional[Callable] = None,
    ) -> AgentState:
        """Run a single execution turn preserving existing conversation state."""
        loop = AgentLoop(
            provider=self.provider,
            executor=self.executor,
            registry=self.registry,
            state=state,
            config=self.config,
            on_step=on_step,
        )
        return await loop.run_task(task)
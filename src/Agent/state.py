"""
Agent state tracking session messages, active tasks, tool execution history, and token usage.
"""

from dataclasses import dataclass, field
from enum import Enum
import uuid
from typing import List, Optional

from src.LLM.base import Message, TokenUsage
from src.tools.base import ToolObservation


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    current_task: str = ""
    current_plan: List[str] = field(default_factory=list)
    observations: List[ToolObservation] = field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    iteration: int = 0
    max_iterations: int = 20
    total_usage: TokenUsage = field(default_factory=TokenUsage)
    last_response_text: Optional[str] = None

    def add_user_message(self, content: str) -> Message:
        msg = Message.user(content)
        self.messages.append(msg)
        return msg

    def add_assistant_message(self, msg: Message) -> None:
        self.messages.append(msg)
        if isinstance(msg.content, str):
            self.last_response_text = msg.content

    def add_observation(self, obs: ToolObservation) -> Message:
        self.observations.append(obs)
        tool_msg = Message.tool(
            content=obs.format_for_llm(),
            tool_call_id=obs.tool_call_id,
            name=obs.tool_name,
        )
        self.messages.append(tool_msg)
        return tool_msg

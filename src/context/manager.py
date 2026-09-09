"""
Context manager for budgeting token consumption and pruning conversation history.
"""

from typing import List
from src.LLM.base import Message, Role
from src.context.tokenizer import count_message_tokens, estimate_tokens


class ContextManager:
    """Manages token window budgeting and message pruning."""

    def __init__(self, max_tokens: int = 100000) -> None:
        self.max_tokens = max_tokens

    def prune_messages(self, messages: List[Message]) -> List[Message]:
        """
        Prune old tool outputs or user messages if total token count exceeds max_tokens.
        Always preserves system instructions (Role.SYSTEM).
        """
        total = count_message_tokens(messages)
        if total <= self.max_tokens:
            return messages

        system_messages = [m for m in messages if m.role == Role.SYSTEM]
        other_messages = [m for m in messages if m.role != Role.SYSTEM]

        # Prune older messages from the beginning of other_messages
        pruned_others = list(other_messages)
        while pruned_others and count_message_tokens(system_messages + pruned_others) > self.max_tokens:
            pruned_others.pop(0)

        return system_messages + pruned_others

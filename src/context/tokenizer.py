"""
Tokenizer estimation helper.
"""

from typing import Union, List, Dict, Any
from src.LLM.base import Message


def estimate_tokens(content: Union[str, List[Dict[str, Any]], None]) -> int:
    """Estimate token length of string or list content."""
    if not content:
        return 0
    if isinstance(content, str):
        return int(len(content.split()) * 1.33) + 1
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, dict) and "text" in item:
                total += int(len(item["text"].split()) * 1.33) + 1
        return total
    return 0


def count_message_tokens(messages: List[Message]) -> int:
    """Calculate total token count across a list of Message objects."""
    total = 0
    for msg in messages:
        total += 4  # message metadata overhead
        total += estimate_tokens(msg.content)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                total += estimate_tokens(tc.name)
                args_str = tc.arguments if isinstance(tc.arguments, str) else str(tc.arguments)
                total += estimate_tokens(args_str)
    return total

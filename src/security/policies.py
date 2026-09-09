"""
Security policy rules and command risk classification.
"""

from enum import Enum
import re


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CommandPolicy:
    """Classifies shell commands into risk levels based on policy rules."""

    LOW_PATTERNS = [
        r"^git\s+(status|diff|log|branch|show)\b",
        r"^(pytest|unittest|pytest-3)\b",
        r"^(ls|dir|cat|grep|rg|find|echo|pwd)\b",
        r"^python\s+-m\s+unittest\b",
        r"^python\s+--version\b",
    ]

    HIGH_PATTERNS = [
        r"\b(rm|del|erase|rd|rmdir)\s+-[a-zA-Z]*[rf]\w*\b",
        r"\bgit\s+reset\s+--hard\b",
        r"\b(drop|truncate|delete)\b",
        r"\bformat\s+[a-z]:\b",
        r"\bchmod\s+777\b",
    ]

    @classmethod
    def classify_command(cls, command: str) -> RiskLevel:
        cmd_str = command.strip().lower()
        for pattern in cls.HIGH_PATTERNS:
            if re.search(pattern, cmd_str):
                return RiskLevel.HIGH

        for pattern in cls.LOW_PATTERNS:
            if re.search(pattern, cmd_str):
                return RiskLevel.LOW

        return RiskLevel.MEDIUM

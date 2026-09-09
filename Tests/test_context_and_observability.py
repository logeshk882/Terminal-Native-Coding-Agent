"""
Unit tests for context manager and observability logging.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path

from src.LLM.base import Message
from src.context.manager import ContextManager
from src.observability.logger import RunLogger


class TestContextAndObservability(unittest.TestCase):

    def test_context_manager_pruning(self):
        cm = ContextManager(max_tokens=30)
        sys_msg = Message.system("System prompt instructions")
        msg1 = Message.user("Short query 1")
        msg2 = Message.user("A very long user query with many words to exceed budget token counts")

        pruned = cm.prune_messages([sys_msg, msg1, msg2])
        self.assertIn(sys_msg, pruned)
        self.assertLessEqual(len(pruned), 3)

    def test_run_logger(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            logger = RunLogger("run_test_123", base_dir=tmp_path)
            logger.log_event("test_event", {"status": "ok"})

            self.assertTrue(logger.log_file.exists())
            content = logger.log_file.read_text(encoding="utf-8")
            self.assertIn("test_event", content)
            self.assertIn("run_test_123", content)


if __name__ == "__main__":
    unittest.main()

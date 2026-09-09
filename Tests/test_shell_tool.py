"""
Unit tests for safe shell execution tool.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path

from src.security.permissions import PermissionManager
from src.tools.builtin.shell import RunCommandArgs, RunCommandTool


class TestShellTool(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.pm = PermissionManager(self.workspace)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_run_command_success(self):
        async def run_test():
            tool = RunCommandTool(self.pm)
            res = await tool.run(RunCommandArgs(command="echo Hello TMCA Shell"))
            self.assertTrue(res.success)
            self.assertIn("Hello TMCA Shell", res.output)
            self.assertEqual(res.exit_code, 0)

        asyncio.run(run_test())

    def test_high_risk_command_rejection(self):
        async def run_test():
            tool = RunCommandTool(self.pm, allow_high_risk=False)
            res = await tool.run(RunCommandArgs(command="rm -rf /"))
            self.assertFalse(res.success)
            self.assertIn("HIGH RISK", res.error)
            self.assertEqual(res.exit_code, 126)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()

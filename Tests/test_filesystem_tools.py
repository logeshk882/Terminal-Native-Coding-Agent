"""
Unit tests for security permissions and filesystem/search tools.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path

from src.security.permissions import PermissionManager
from src.tools.builtin.filesystem import (
    ApplyPatchArgs,
    ApplyPatchTool,
    ListDirectoryArgs,
    ListDirectoryTool,
    ReadFileArgs,
    ReadFileTool,
    WriteFileArgs,
    WriteFileTool,
)
from src.tools.builtin.search import SearchFilesArgs, SearchFilesTool


class TestFilesystemTools(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.pm = PermissionManager(self.workspace)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_permission_manager_path_traversal_protection(self):
        # Valid path
        valid = self.pm.validate_path("src/main.py")
        self.assertTrue(str(valid).startswith(str(self.workspace)))

        # Invalid path traversal attempt
        with self.assertRaises(PermissionError):
            self.pm.validate_path("../../etc/passwd")

    def test_write_read_and_patch(self):
        async def run_test():
            write_tool = WriteFileTool(self.pm)
            read_tool = ReadFileTool(self.pm)
            patch_tool = ApplyPatchTool(self.pm)
            list_tool = ListDirectoryTool(self.pm)

            # Write file
            w_res = await write_tool.run(WriteFileArgs(path="hello.txt", content="Hello World!\nSecond Line"))
            self.assertTrue(w_res.success)

            # Read file
            r_res = await read_tool.run(ReadFileArgs(path="hello.txt"))
            self.assertTrue(r_res.success)
            self.assertIn("Hello World!", r_res.output)

            # List directory
            l_res = await list_tool.run(ListDirectoryArgs(path="."))
            self.assertTrue(l_res.success)
            self.assertIn("hello.txt", l_res.output)

            # Apply patch
            p_res = await patch_tool.run(
                ApplyPatchArgs(path="hello.txt", target_content="World", replacement_content="TMCA Agent")
            )
            self.assertTrue(p_res.success)

            r_res2 = await read_tool.run(ReadFileArgs(path="hello.txt"))
            self.assertIn("Hello TMCA Agent!", r_res2.output)

        asyncio.run(run_test())

    def test_search_files(self):
        async def run_test():
            write_tool = WriteFileTool(self.pm)
            await write_tool.run(WriteFileArgs(path="sub/code.py", content="def calculate_sum(a, b):\n    return a + b"))

            search_tool = SearchFilesTool(self.pm)
            res = await search_tool.run(SearchFilesArgs(query="calculate_sum", path="."))
            self.assertTrue(res.success)
            self.assertIn("calculate_sum", res.output)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()

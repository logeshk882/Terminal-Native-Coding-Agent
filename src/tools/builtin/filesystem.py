"""
Filesystem built-in tools for listing, reading, writing, and patching files.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

from src.security.permissions import PermissionManager
from src.tools.base import Tool, ToolObservation


class ListDirectoryArgs(BaseModel):
    path: str = Field(default=".", description="Relative path of directory to list")


class ListDirectoryTool(Tool):

    def __init__(self, permission_manager: PermissionManager) -> None:
        self.permission_manager = permission_manager

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return "List files and subdirectories in a directory relative to the workspace."

    @property
    def args_model(self):
        return ListDirectoryArgs

    async def run(self, args: ListDirectoryArgs) -> ToolObservation:
        target = self.permission_manager.validate_path(args.path)
        if not target.exists():
            return ToolObservation(
                tool_name=self.name,
                tool_call_id="",
                success=False,
                output="",
                error=f"Directory '{args.path}' does not exist.",
            )
        if not target.is_dir():
            return ToolObservation(
                tool_name=self.name,
                tool_call_id="",
                success=False,
                output="",
                error=f"Path '{args.path}' is not a directory.",
            )

        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = []
        for entry in entries:
            kind = "[DIR]" if entry.is_dir() else "[FILE]"
            rel = entry.relative_to(self.permission_manager.workspace_dir)
            lines.append(f"{kind:<6} {rel}")

        return ToolObservation(
            tool_name=self.name,
            tool_call_id="",
            success=True,
            output="\n".join(lines) if lines else "Directory is empty.",
        )


class ReadFileArgs(BaseModel):
    path: str = Field(description="Relative path of file to read")
    start_line: Optional[int] = Field(default=None, description="Optional 1-based start line number")
    end_line: Optional[int] = Field(default=None, description="Optional 1-based end line number")


class ReadFileTool(Tool):

    def __init__(self, permission_manager: PermissionManager) -> None:
        self.permission_manager = permission_manager

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read UTF-8 text file content with optional line number bounds."

    @property
    def args_model(self):
        return ReadFileArgs

    async def run(self, args: ReadFileArgs) -> ToolObservation:
        target = self.permission_manager.validate_path(args.path)
        if not target.exists():
            return ToolObservation(
                tool_name=self.name,
                tool_call_id="",
                success=False,
                output="",
                error=f"File '{args.path}' does not exist.",
            )

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return ToolObservation(
                tool_name=self.name,
                tool_call_id="",
                success=False,
                output="",
                error=f"Failed to read file '{args.path}': {e}",
            )

        lines = content.splitlines()
        start = (args.start_line - 1) if args.start_line and args.start_line > 0 else 0
        end = args.end_line if args.end_line and args.end_line <= len(lines) else len(lines)

        selected = lines[start:end]
        formatted = "\n".join(f"{i + start + 1:4d} | {line}" for i, line in enumerate(selected))

        return ToolObservation(
            tool_name=self.name,
            tool_call_id="",
            success=True,
            output=formatted,
        )


class WriteFileArgs(BaseModel):
    path: str = Field(description="Relative path of file to write")
    content: str = Field(description="Content to write to file")


class WriteFileTool(Tool):

    def __init__(self, permission_manager: PermissionManager) -> None:
        self.permission_manager = permission_manager

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write text content to a file in the workspace, creating directories if needed."

    @property
    def args_model(self):
        return WriteFileArgs

    async def run(self, args: WriteFileArgs) -> ToolObservation:
        target = self.permission_manager.validate_path(args.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.content, encoding="utf-8")

        return ToolObservation(
            tool_name=self.name,
            tool_call_id="",
            success=True,
            output=f"Successfully wrote {len(args.content)} bytes to {args.path}.",
        )


class ApplyPatchArgs(BaseModel):
    path: str = Field(description="Relative path of file to patch")
    target_content: str = Field(description="Exact substring content to replace")
    replacement_content: str = Field(description="Replacement content")


class ApplyPatchTool(Tool):

    def __init__(self, permission_manager: PermissionManager) -> None:
        self.permission_manager = permission_manager

    @property
    def name(self) -> str:
        return "apply_patch"

    @property
    def description(self) -> str:
        return "Replace target_content with replacement_content in a file."

    @property
    def args_model(self):
        return ApplyPatchArgs

    async def run(self, args: ApplyPatchArgs) -> ToolObservation:
        target = self.permission_manager.validate_path(args.path)
        if not target.exists():
            return ToolObservation(
                tool_name=self.name,
                tool_call_id="",
                success=False,
                output="",
                error=f"File '{args.path}' does not exist.",
            )

        content = target.read_text(encoding="utf-8")
        if args.target_content not in content:
            return ToolObservation(
                tool_name=self.name,
                tool_call_id="",
                success=False,
                output="",
                error=f"Target content not found in '{args.path}'.",
            )

        count = content.count(args.target_content)
        if count > 1:
            return ToolObservation(
                tool_name=self.name,
                tool_call_id="",
                success=False,
                output="",
                error=f"Target content matched {count} times in '{args.path}'. Expected exactly 1 match for safety.",
            )

        new_content = content.replace(args.target_content, args.replacement_content, 1)
        target.write_text(new_content, encoding="utf-8")

        return ToolObservation(
            tool_name=self.name,
            tool_call_id="",
            success=True,
            output=f"Successfully applied patch to {args.path}.",
        )

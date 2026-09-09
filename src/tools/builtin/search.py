"""
Search files tool using ripgrep for fast code search.
"""

import asyncio
import shutil
from typing import Optional
from pydantic import BaseModel, Field

from src.security.permissions import PermissionManager
from src.tools.base import Tool, ToolObservation


class SearchFilesArgs(BaseModel):
    query: str = Field(description="Search string or regex pattern")
    path: str = Field(default=".", description="Relative path or subfolder to search within")


class SearchFilesTool(Tool):

    def __init__(self, permission_manager: PermissionManager) -> None:
        self.permission_manager = permission_manager

    @property
    def name(self) -> str:
        return "search_files"

    @property
    def description(self) -> str:
        return "Search for text or regex pattern across workspace files."

    @property
    def args_model(self):
        return SearchFilesArgs

    async def run(self, args: SearchFilesArgs) -> ToolObservation:
        target = self.permission_manager.validate_path(args.path)

        rg_path = shutil.which("rg")
        if rg_path:
            cmd = [rg_path, "--line-number", "--color=never", "--max-columns=200", args.query, str(target)]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.permission_manager.workspace_dir),
                )
                stdout, stderr = await proc.communicate()
                out_text = stdout.decode("utf-8", errors="replace").strip()
                if not out_text:
                    return ToolObservation(
                        tool_name=self.name,
                        tool_call_id="",
                        success=True,
                        output=f"No matches found for query '{args.query}'.",
                    )
                return ToolObservation(
                    tool_name=self.name,
                    tool_call_id="",
                    success=True,
                    output=out_text[:10000],  # Limit output length
                )
            except Exception as e:
                pass

        # Python naive fallback if ripgrep is unavailable
        matches = []
        for file_path in target.rglob("*"):
            if file_path.is_file() and not any(part.startswith(".") or part == "node_modules" for part in file_path.parts):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    for i, line in enumerate(content.splitlines(), start=1):
                        if args.query.lower() in line.lower():
                            rel = file_path.relative_to(self.permission_manager.workspace_dir)
                            matches.append(f"{rel}:{i}:{line.strip()}")
                            if len(matches) >= 100:
                                break
                except Exception:
                    continue
            if len(matches) >= 100:
                break

        if not matches:
            return ToolObservation(
                tool_name=self.name,
                tool_call_id="",
                success=True,
                output=f"No matches found for '{args.query}'.",
            )

        return ToolObservation(
            tool_name=self.name,
            tool_call_id="",
            success=True,
            output="\n".join(matches),
        )

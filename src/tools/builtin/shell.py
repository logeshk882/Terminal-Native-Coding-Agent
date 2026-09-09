"""
Shell execution tool for running safe terminal commands within the workspace.
"""

import asyncio
import time
from typing import Optional
from pydantic import BaseModel, Field

from src.security.permissions import PermissionManager
from src.security.policies import CommandPolicy, RiskLevel
from src.tools.base import Tool, ToolObservation


class RunCommandArgs(BaseModel):
    command: str = Field(description="Terminal shell command string to execute")
    timeout: float = Field(default=30.0, description="Maximum execution timeout in seconds")


class RunCommandTool(Tool):

    def __init__(self, permission_manager: PermissionManager, allow_high_risk: bool = False) -> None:
        self.permission_manager = permission_manager
        self.allow_high_risk = allow_high_risk

    @property
    def name(self) -> str:
        return "run_command"

    @property
    def description(self) -> str:
        return "Execute a shell command (e.g. pytest, git status, python) in the workspace."

    @property
    def args_model(self):
        return RunCommandArgs

    async def run(self, args: RunCommandArgs) -> ToolObservation:
        risk = CommandPolicy.classify_command(args.command)
        if risk == RiskLevel.HIGH and not self.allow_high_risk:
            return ToolObservation(
                tool_name=self.name,
                tool_call_id="",
                success=False,
                output="",
                exit_code=126,
                error=f"Command '{args.command}' classified as HIGH RISK and rejected by security policy.",
                raw_data={"risk_level": risk.value},
            )

        start_time = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_shell(
                args.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.permission_manager.workspace_dir),
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=args.timeout)
                duration = round((time.perf_counter() - start_time) * 1000, 2)
                stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
                stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

                combined = stdout
                if stderr:
                    combined = f"{stdout}\n--- STDERR ---\n{stderr}" if stdout else stderr

                return ToolObservation(
                    tool_name=self.name,
                    tool_call_id="",
                    success=(proc.returncode == 0),
                    output=combined,
                    exit_code=proc.returncode,
                    duration_ms=duration,
                    raw_data={
                        "command": args.command,
                        "risk_level": risk.value,
                        "stdout": stdout,
                        "stderr": stderr,
                    },
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                duration = round((time.perf_counter() - start_time) * 1000, 2)
                return ToolObservation(
                    tool_name=self.name,
                    tool_call_id="",
                    success=False,
                    output="",
                    exit_code=124,
                    duration_ms=duration,
                    error=f"Command '{args.command}' timed out after {args.timeout} seconds.",
                    raw_data={"risk_level": risk.value, "timed_out": True},
                )
        except Exception as e:
            duration = round((time.perf_counter() - start_time) * 1000, 2)
            return ToolObservation(
                tool_name=self.name,
                tool_call_id="",
                success=False,
                output="",
                exit_code=1,
                duration_ms=duration,
                error=f"Failed to execute command '{args.command}': {str(e)}",
            )

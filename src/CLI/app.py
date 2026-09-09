"""
Typer CLI application entry point with Rich terminal UI rendering.
"""

import asyncio
import os
from pathlib import Path
import sys
from typing import Optional
from dotenv import load_dotenv
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status

from src.Agent.agent import TMCAAgent
from src.Agent.state import AgentState, AgentStatus
from src.LLM.base import LLMConfig
from src.LLM.fallback import FallbackLLMProvider
from src.LLM.omniroute import OmniRouteProvider
from src.LLM.openAI import OpenAIProvider
from src.LLM.openrouter import OpenRouterProvider
from src.observability.logger import RunLogger
from src.tools.base import ToolObservation

# Automatically load environment variables from .env file
load_dotenv()

# Force UTF-8 stdout/stderr encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

cli = typer.Typer(name="tmca", help="TMCA - Terminal-Native Coding Assistant")
console = Console(legacy_windows=False)


def _build_provider_by_name(name: str, config: LLMConfig):
    n = name.strip().lower()
    if n == "openrouter":
        return OpenRouterProvider(config=config)
    if n == "omniroute":
        return OmniRouteProvider(config=config)
    if n == "openai":
        return OpenAIProvider(config=config)
    return None


def _get_provider(provider_name: str, config: LLMConfig):
    name = provider_name.lower().strip()

    # 1. Comma-separated list passed via CLI (e.g. --provider openrouter,openai,omniroute)
    if "," in name:
        parts = [p.strip() for p in name.split(",") if p.strip()]
        chain = []
        for p in parts:
            prov = _build_provider_by_name(p, config)
            if prov:
                chain.append(prov)
        if chain:
            return FallbackLLMProvider(providers=chain, config=config)

    # 2. Single provider explicitly requested via CLI (e.g. --provider openrouter)
    single = _build_provider_by_name(name, config)
    if single and name != "auto":
        return single

    # 3. Custom provider priority order set in .env (e.g. PROVIDER_ORDER=openrouter,openai,omniroute)
    env_order = os.getenv("PROVIDER_ORDER") or os.getenv("TMCA_PROVIDER_ORDER")
    if env_order:
        parts = [p.strip() for p in env_order.split(",") if p.strip()]
        chain = []
        for p in parts:
            prov = _build_provider_by_name(p, config)
            if prov:
                chain.append(prov)
        if chain:
            return FallbackLLMProvider(providers=chain, config=config)

    # 4. Default provider failover priority
    available_providers = []
    if os.getenv("OPENROUTER_API_KEY"):
        available_providers.append(OpenRouterProvider(config=config))
    if os.getenv("OPENAI_API_KEY"):
        available_providers.append(OpenAIProvider(config=config))
    if os.getenv("OMNIROUTE_API_KEY"):
        available_providers.append(OmniRouteProvider(config=config))

    if not available_providers:
        available_providers = [
            OpenRouterProvider(config=config),
            OpenAIProvider(config=config),
            OmniRouteProvider(config=config),
        ]

    return FallbackLLMProvider(providers=available_providers, config=config)


@cli.command()
def main(
    task: Optional[str] = typer.Argument(None, help="Initial natural language coding task or question"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
    provider: str = typer.Option("auto", "--provider", "-p", help="Provider name: 'auto', 'openai', 'openrouter', or 'omniroute'"),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Workspace root directory"),
    allow_high_risk: bool = typer.Option(False, "--allow-high-risk", help="Allow execution of high-risk commands"),
):
    """Continuous interactive REPL session for TMCA Agent."""
    console.print(
        Panel(
            f"[bold green]TMCA Interactive Session[/bold green]\n"
            f"[dim]Model: {model} | Provider: {provider}[/dim]\n"
            f"[yellow]Type 'stop', 'exit', or 'quit' anytime to end the session.[/yellow]",
            title="TMCA Agent Initialized",
        )
    )

    cfg = LLMConfig(model_name=model)
    llm_provider = _get_provider(provider, cfg)
    agent = TMCAAgent(
        provider=llm_provider,
        workspace_dir=workspace,
        config=cfg,
        allow_high_risk_commands=allow_high_risk,
    )

    logger = RunLogger(run_id=agent.permission_manager.workspace_dir.name)
    state = agent.create_state()

    def step_callback(st: AgentState, obs: Optional[ToolObservation]):
        if obs:
            logger.log_event("tool_observation", obs.to_dict())
            if obs.success:
                console.print(f"  [bold blue]Tool Call ({obs.tool_name}):[/bold blue] {obs.output[:120]}...")
            else:
                console.print(f"  [bold red]Tool Failure ({obs.tool_name}):[/bold red] {obs.error}")

    current_task = task
    while True:
        if not current_task:
            try:
                current_task = typer.prompt("you").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[bold yellow]Session ended. Goodbye![/bold yellow]")
                break

        if current_task.lower() in ("stop", "exit", "quit"):
            console.print("\n[bold yellow]Session ended. Goodbye![/bold yellow]")
            break

        if not current_task:
            continue

        async def _run(prompt_str: str):
            with Status("[bold yellow]Agent working...", console=console):
                return await agent.run_turn(state, prompt_str, on_step=step_callback)

        try:
            asyncio.run(_run(current_task))
            logger.log_event("turn_finish", {"status": state.status.value, "total_usage": state.total_usage.to_dict()})

            console.print("\n" + "=" * 60 + "\n")
            if state.last_response_text:
                console.print(Markdown(state.last_response_text))
            
            console.print(f"\n[dim]Iteration {state.iteration} | Total Tokens: {state.total_usage.total_tokens}[/dim]")
        except Exception as e:
            console.print(f"\n[bold red]Error executing turn:[/bold red] {e}")

        current_task = None


if __name__ == "__main__":
    cli()

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer

from .browser.sandbox import SandboxPolicy
from .config import Settings
from .llm.anthropic_client import AnthropicClient
from .llm.openai_client import OpenAIClient
from .logging import setup_logging
from .core.agent import Agent
from .core.planner import Planner

app = typer.Typer(no_args_is_help=True)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        del sys.argv[1]
    app()


@app.command()
def run(
    task: str = typer.Option(..., help="Task description for the agent"),
    start_url: Optional[str] = typer.Option(None, help="Initial URL to open"),
    provider: Optional[str] = typer.Option(None, help="LLM provider override"),
    headful: bool = typer.Option(False, help="Run browser in headed mode"),
    allowlist: Optional[str] = typer.Option(None, help="Comma separated host allowlist"),
    max_steps: Optional[int] = typer.Option(None, help="Override for maximum steps"),
    timeout: Optional[int] = typer.Option(None, help="Override for run timeout (seconds)"),
    allow_js_eval: bool = typer.Option(False, help="Allow JavaScript evaluation"),
    allow_file_upload: bool = typer.Option(False, help="Allow file upload actions"),
    browser_profile_dir: Optional[Path] = typer.Option(
        None,
        help="Directory to store a persistent browser profile (reuse cookies and sessions)",
    ),
) -> None:
    settings = Settings.from_env()
    if provider:
        settings.llm_provider = provider  # type: ignore[assignment]
    if allowlist:
        settings.update_allowlist(allowlist.split(","))
    if max_steps:
        settings.max_steps = max_steps
    if timeout:
        settings.run_timeout_s = timeout
    settings.allow_js_eval = allow_js_eval
    settings.allow_file_upload = allow_file_upload
    if browser_profile_dir is not None:
        settings.browser_profile_dir = browser_profile_dir.expanduser()
    settings.ensure_directories()
    setup_logging(settings.log_level, settings.log_dir / "agent.log")

    headless = not headful
    asyncio.run(_run(task, start_url, settings, headless=headless))


async def _run(task: str, start_url: Optional[str], settings: Settings, headless: bool) -> None:
    sandbox = SandboxPolicy.from_hosts(
        hosts=settings.allowlist_hosts,
        max_steps=settings.max_steps,
        step_timeout_s=settings.step_timeout_s,
        run_timeout_s=settings.run_timeout_s,
        allow_js_eval=settings.allow_js_eval,
        allow_file_upload=settings.allow_file_upload,
    )

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise typer.BadParameter("OPENAI_API_KEY not configured")
        llm_client = OpenAIClient(settings.openai_api_key)
    else:
        if not settings.anthropic_api_key:
            raise typer.BadParameter("ANTHROPIC_API_KEY not configured")
        llm_client = AnthropicClient(settings.anthropic_api_key)

    planner = Planner(llm_client)
    agent = Agent(
        planner=planner,
        settings=settings,
        sandbox=sandbox,
        runs_dir=settings.runs_dir,
        capture_screenshots=True,
    )

    result = await agent.run(task=task, start_url=start_url, headless=headless)
    typer.echo(f"Run status: {result.status}")
    if result.final_answer:
        typer.echo(f"Final answer: {result.final_answer}")
    if result.error:
        typer.echo(f"Error: {result.error}")
    typer.echo(f"Steps executed: {result.total_steps()}")

    if result.steps:
        typer.echo("Step details:")
        for step in result.steps:
            typer.echo(
                f"  #{step.step_index} {step.action.type} – {step.action.reason}"
            )
            typer.echo(f"     Result: {step.result_summary}")
            if step.action.selector:
                typer.echo(f"     Selector: {step.action.selector}")

    if hasattr(llm_client, "close"):
        await getattr(llm_client, "close")()

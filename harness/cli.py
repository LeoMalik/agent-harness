from __future__ import annotations

from pathlib import Path

import typer

from harness.config import Config
from harness.runtime import Runtime
from harness.types import TurnStatus

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def chat(
    text: str = typer.Argument(..., help="User message"),
    session: str | None = typer.Option(None, "--session", "-s"),
    workspace: str = typer.Option("default", "--workspace", "-w"),
    user: str = typer.Option("local", "--user", "-u"),
    cwd: Path = typer.Option(Path("."), "--cwd"),
) -> None:
    runtime = Runtime.create(Config(), user_id=user, workspace_id=workspace, cwd=cwd, session_id=session)
    turn = runtime.run(text)
    _print_turn(runtime, turn)


@app.command()
def resume(
    session: str = typer.Option(..., "--session", "-s"),
    answer: str = typer.Argument(..., help="Human answer for a pending ask_user call"),
    cwd: Path = typer.Option(Path("."), "--cwd"),
) -> None:
    runtime = Runtime.create(Config(), cwd=cwd, session_id=session)
    turn = runtime.resume(session, answer)
    _print_turn(runtime, turn)


@app.command()
def cancel(
    session: str = typer.Option(..., "--session", "-s"),
    cwd: Path = typer.Option(Path("."), "--cwd"),
) -> None:
    runtime = Runtime.create(Config(), cwd=cwd, session_id=session)
    turn = runtime.history.load_turn(session)
    if turn is None:
        raise typer.BadParameter("no turn to cancel")
    runtime.cancel(turn.turn_id)
    latest = runtime.history.load_turn(session) or turn
    _print_turn(runtime, latest)


def _print_turn(runtime: Runtime, turn) -> None:
    typer.echo(f"session={runtime.session.session_id}")
    typer.echo(f"turn={turn.turn_id} status={turn.status}")
    if turn.status == TurnStatus.PENDING.value:
        typer.echo(f"waiting_for={turn.waiting_for} wait_ids={','.join(turn.wait_ids)}")
        if turn.resume_token:
            typer.echo(f"resume_token={turn.resume_token}")
    if turn.final_text:
        typer.echo(turn.final_text)

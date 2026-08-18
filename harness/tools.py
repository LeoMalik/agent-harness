from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from harness.config import Config
from harness.types import Observation, ToolCall, new_id


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[..., Observation]
    needs_approval: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def default_tools(config: Config, cwd: Path) -> dict[str, Tool]:
    cwd = cwd.resolve()

    def read_file(path: str, **_: Any) -> Observation:
        target = _resolve(cwd, path)
        text = target.read_text(encoding="utf-8")
        return _maybe_artifact(config, "read_file", text, refs=[str(target)])

    def write_file(path: str, content: str, **_: Any) -> Observation:
        target = _resolve(cwd, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return Observation(
            tool_call_id="",
            tool_name="write_file",
            summary=f"Wrote {len(content)} chars to {target}",
            refs=[str(target)],
        )

    def bash(command: str, **_: Any) -> Observation:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=config.tool_timeout,
        )
        text = (completed.stdout or "") + (completed.stderr or "")
        summary = text.strip() or f"exit {completed.returncode}"
        observation = _maybe_artifact(config, "bash", summary, refs=[str(cwd)])
        if completed.returncode != 0:
            observation.outcome = "fail"
            observation.error = f"exit {completed.returncode}"
        return observation

    def search_web(query: str, **_: Any) -> Observation:
        if not config.tavily_api_key:
            return Observation(
                tool_call_id="",
                tool_name="search_web",
                outcome="fail",
                summary="TAVILY_API_KEY is not set",
                error="missing_tavily_key",
            )
        with httpx.Client(timeout=30) as client:
            response = client.post(
                "https://api.tavily.com/search",
                json={"api_key": config.tavily_api_key, "query": query, "max_results": 5},
            )
            response.raise_for_status()
            data = response.json()
        lines = []
        refs = []
        for item in data.get("results") or []:
            title = item.get("title") or ""
            url = item.get("url") or ""
            snippet = item.get("content") or ""
            lines.append(f"- {title}\n  {url}\n  {snippet}")
            if url:
                refs.append(url)
        return Observation(
            tool_call_id="",
            tool_name="search_web",
            summary="\n".join(lines) or "No results",
            refs=refs,
        )

    def remember(slot: str, text: str, layer: str = "profile", **_: Any) -> Observation:
        return Observation(
            tool_call_id="",
            tool_name="remember",
            summary=f"{slot} = {text}",
            refs=[slot],
            preview=json.dumps({"slot": slot, "text": text, "layer": layer}),
        )

    def ask_user(question: str, options: list[str] | None = None, **_: Any) -> Observation:
        return Observation(
            tool_call_id="",
            tool_name="ask_user",
            summary=question,
            preview=json.dumps({"question": question, "options": options or []}),
        )

    tools = [
        Tool(
            "read_file",
            "Read a UTF-8 text file.",
            _object({"path": _string("Path to read")}),
            read_file,
        ),
        Tool(
            "write_file",
            "Write a UTF-8 text file. Ask before overwriting important files.",
            _object({"path": _string("Path to write"), "content": _string("File content")}),
            write_file,
            needs_approval=True,
        ),
        Tool(
            "bash",
            "Run a shell command in the workspace.",
            _object({"command": _string("Shell command")}),
            bash,
            needs_approval=True,
        ),
        Tool(
            "search_web",
            "Search the web with Tavily.",
            _object({"query": _string("Search query")}),
            search_web,
        ),
        Tool(
            "remember",
            "Store a durable personal fact. Use slot keys like work.city.",
            _object(
                {
                    "slot": _string("Stable slot key"),
                    "text": _string("Fact to remember"),
                    "layer": {"type": "string", "enum": ["profile", "rag"], "description": "profile if needed almost every turn"},
                },
                required=["slot", "text"],
            ),
            remember,
        ),
        Tool(
            "ask_user",
            "Ask the user a question and wait.",
            _object(
                {
                    "question": _string("Question to ask"),
                    "options": {"type": "array", "items": {"type": "string"}, "description": "Optional choices"},
                },
                required=["question"],
            ),
            ask_user,
        ),
    ]
    return {tool.name: tool for tool in tools}


def spawn_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "spawn",
            "description": "Spawn a child agent for an isolated subtask. Use templates general, explore, or test.",
            "parameters": _object(
                {
                    "template": {"type": "string", "enum": ["general", "explore", "test"]},
                    "goal": _string("What the child must finish"),
                    "graph": _string("Optional saved graph name to follow"),
                },
                required=["template", "goal"],
            ),
        },
    }


def _maybe_artifact(config: Config, tool_name: str, text: str, refs: list[str]) -> Observation:
    if len(text) <= config.max_inline_chars:
        return Observation(tool_call_id="", tool_name=tool_name, summary=text, refs=refs)
    artifact_id = new_id("art")
    preview = text[:500]
    db = config.db()
    if db:
        db.insert(
            "artifacts",
            {"artifact_id": artifact_id, "content": text, "preview": preview},
        )
        location = f"supabase://artifacts/{artifact_id}"
    else:
        path = config.data_dir / "artifacts" / f"{artifact_id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        location = str(path)
    return Observation(
        tool_call_id="",
        tool_name=tool_name,
        summary=f"Output stored as artifact {artifact_id} ({len(text)} chars). Read {location} if more is needed.",
        refs=refs + [location],
        artifact_id=artifact_id,
        preview=preview,
    )


def _resolve(cwd: Path, path: str) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = cwd / target
    return target.resolve()


def _string(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or list(properties),
        "additionalProperties": False,
    }


def bind(observation: Observation, call: ToolCall) -> Observation:
    observation.tool_call_id = call.tool_call_id
    return observation

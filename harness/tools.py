from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from harness.config import Config
from harness.hooks import Hook, approval_hook, idempotency_hooks, interrupt_hook
from harness.types import Observation, ToolCall, new_id


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[..., Observation]
    needs_approval: bool = False
    before_hooks: list[Hook] = field(default_factory=list)
    after_hooks: list[Hook] = field(default_factory=list)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def default_tools(config: Config, cwd: Path, user_id: str = "local") -> dict[str, Tool]:
    cwd = cwd.resolve()

    def read_file(path: str, **_: Any) -> Observation:
        db = config.db()
        if db is None:
            raise RuntimeError("Supabase is required for file storage")
        key = _object_path(user_id, path)
        text = db.storage_get(config.storage_bucket, key)
        return _maybe_artifact(config, "read_file", text, refs=[f"supabase://storage/{config.storage_bucket}/{key}"])

    def write_file(path: str, content: str, **_: Any) -> Observation:
        db = config.db()
        if db is None:
            raise RuntimeError("Supabase is required for file storage")
        key = _object_path(user_id, path)
        db.storage_put(config.storage_bucket, key, content)
        return Observation(
            tool_call_id="",
            tool_name="write_file",
            summary=f"Wrote {len(content)} chars to {key}",
            refs=[f"supabase://storage/{config.storage_bucket}/{key}"],
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

    def ask_user(question: str, options: list[str] | None = None, _human: dict[str, Any] | None = None, **_: Any) -> Observation:
        answer = (_human or {}).get("answer")
        if answer is not None and str(answer).strip() != "":
            return Observation(
                tool_call_id="",
                tool_name="ask_user",
                summary=str(answer),
                preview=json.dumps({"question": question, "options": options or [], "answer": answer}),
            )
        return Observation(
            tool_call_id="",
            tool_name="ask_user",
            summary=question,
            preview=json.dumps({"question": question, "options": options or []}),
        )

    def read_artifact_range(artifact_id: str, start_byte: int = 0, end_byte: int | None = None, **_: Any) -> Observation:
        db = config.db()
        if db is None:
            raise RuntimeError("Supabase is required for artifact storage")
        rows = db.select("artifacts", {"select": "*", "artifact_id": f"eq.{artifact_id}", "limit": "1"})
        if not rows:
            return Observation(
                tool_call_id="",
                tool_name="read_artifact_range",
                outcome="fail",
                summary=f"Artifact {artifact_id} not found",
                error="not_found",
            )
        text = str(rows[0].get("content") or "")
        start = max(0, int(start_byte or 0))
        limit = config.artifact_range_bytes
        end = min(len(text), int(end_byte) + 1 if end_byte is not None else start + limit)
        if end - start > limit:
            end = start + limit
        chunk = text[start:end]
        return Observation(
            tool_call_id="",
            tool_name="read_artifact_range",
            summary=chunk,
            refs=[f"supabase://artifacts/{artifact_id}"],
            artifact_id=artifact_id,
            preview=f"bytes {start}-{max(start, end - 1)} of {len(text)}",
        )

    write_before, write_after = idempotency_hooks()
    bash_before, bash_after = idempotency_hooks()
    tools = [
        Tool(
            "read_file",
            "Read a UTF-8 text file from persistent storage (Supabase Storage).",
            _object({"path": _string("Path to read")}),
            read_file,
        ),
        Tool(
            "write_file",
            "Write a UTF-8 text file to persistent storage (Supabase Storage).",
            _object({"path": _string("Path to write"), "content": _string("File content")}),
            write_file,
            needs_approval=True,
            before_hooks=[approval_hook(), write_before],
            after_hooks=[write_after],
        ),
        Tool(
            "bash",
            "Run a shell command in the workspace.",
            _object({"command": _string("Shell command")}),
            bash,
            needs_approval=True,
            before_hooks=[approval_hook(), bash_before],
            after_hooks=[bash_after],
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
            before_hooks=[interrupt_hook()],
        ),
        Tool(
            "read_artifact_range",
            "Read a byte range from a stored artifact. Use after a tool returns artifact_id.",
            _object(
                {
                    "artifact_id": _string("Artifact id from a previous observation"),
                    "start_byte": {"type": "integer", "description": "Inclusive start offset", "default": 0},
                    "end_byte": {"type": "integer", "description": "Inclusive end offset; capped by Runtime"},
                },
                required=["artifact_id"],
            ),
            read_artifact_range,
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
    if db is None:
        raise RuntimeError("Supabase is required for artifact storage")
    db.insert(
        "artifacts",
        {"artifact_id": artifact_id, "content": text, "preview": preview},
    )
    location = f"supabase://artifacts/{artifact_id}"
    return Observation(
        tool_call_id="",
        tool_name=tool_name,
        summary=f"Output stored as artifact {artifact_id} ({len(text)} chars). Read {location} if more is needed.",
        refs=refs + [location],
        artifact_id=artifact_id,
        preview=preview,
    )


def _storage_path(path: str) -> str:
    parts = [seg for seg in path.strip().replace("\\", "/").split("/") if seg and seg not in (".", "..")]
    return "/".join(parts)


def _object_path(user_id: str, path: str) -> str:
    p = _storage_path(path)
    if p.startswith("skills/"):
        return f"{user_id}/skills/{p[len('skills/'):]}"
    if p.startswith("graphs/"):
        return f"{user_id}/graphs/{p[len('graphs/'):]}"
    return f"{user_id}/files/{p}"


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

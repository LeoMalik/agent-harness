from __future__ import annotations

from pathlib import Path

import pytest

from harness.config import Config
from harness.context import ContextBuilder, USER_TAG
from harness.history import History
from harness.memory import Memory
from harness.runtime import Runtime
from harness.store import MemoryStore, is_cancelled, request_cancel
from harness.tools import default_tools
from harness.types import Event, ModelResponse, ToolCall, TurnStatus


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools=None, temperature=0.2, on_delta=None):
        self.calls.append(messages)
        if not self.responses:
            return ModelResponse(text="empty")
        response = self.responses.pop(0)
        if on_delta and response.thinking:
            on_delta("thinking", response.thinking)
        if on_delta and response.text:
            on_delta("assistant", response.text)
        return response


class FakeDB:
    """In-memory stand-in for harness.db.Supabase (select/insert/upsert/patch/ping)."""

    def __init__(self):
        self.tables: dict[str, list[dict]] = {}

    def _rows(self, table: str) -> list[dict]:
        return self.tables.setdefault(table, [])

    def select(self, table: str, params: dict) -> list[dict]:
        result = [dict(row) for row in self._rows(table)]
        for key, value in params.items():
            if key in ("select", "order", "limit", "offset"):
                continue
            if isinstance(value, str) and value.startswith("eq."):
                wanted = value[3:]
                result = [row for row in result if str(row.get(key)) == wanted]
        order = params.get("order")
        if order:
            column = order.split(".")[0]
            reverse = order.endswith(".desc")
            result.sort(key=lambda row: str(row.get(column, "")), reverse=reverse)
        limit = params.get("limit")
        if limit:
            result = result[: int(limit)]
        return result

    def insert(self, table: str, row: dict) -> list[dict]:
        copied = dict(row)
        self._rows(table).append(copied)
        return [copied]

    def upsert(self, table: str, row: dict, on_conflict: str) -> list[dict]:
        rows = self._rows(table)
        key = row.get(on_conflict)
        for index, existing in enumerate(rows):
            if existing.get(on_conflict) == key:
                rows[index] = dict(row)
                return [dict(row)]
        rows.append(dict(row))
        return [dict(row)]

    def patch(self, table: str, params: dict, row: dict) -> list[dict]:
        return [dict(row)]

    def ping(self) -> bool:
        return True


def make_config(tmp_path: Path) -> Config:
    config = Config(
        root=tmp_path,
        prompts_dir=tmp_path / "prompts",
        skills_dir=tmp_path / "skills",
        graphs_dir=tmp_path / "graphs",
    )
    (tmp_path / "prompts").mkdir(exist_ok=True)
    (tmp_path / "prompts" / "AGENTS.md").write_text("Follow the user.", encoding="utf-8")
    (tmp_path / "prompts" / "SOUL.md").write_text("Be brief.", encoding="utf-8")
    return config


def make_runtime(tmp_path: Path, responses: list, monkeypatch: pytest.MonkeyPatch) -> Runtime:
    db = FakeDB()
    config = make_config(tmp_path)
    monkeypatch.setattr(Config, "db", lambda self: db)
    runtime = Runtime.create(config, cwd=tmp_path, store=MemoryStore())
    runtime.llm = FakeLLM(responses)
    return runtime


def test_direct_answer(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path, [ModelResponse(text="hello")], monkeypatch)
    turn = runtime.run("hi")
    assert turn.status == TurnStatus.COMPLETED.value
    assert turn.final_text == "hello"
    assert runtime.history.load_session(runtime.session.session_id) is not None


def test_read_file_tool(tmp_path, monkeypatch):
    (tmp_path / "note.txt").write_text("alpha", encoding="utf-8")
    runtime = make_runtime(
        tmp_path,
        [
            ModelResponse(tool_calls=[ToolCall("c1", "read_file", {"path": "note.txt"})]),
            ModelResponse(text="the file says alpha"),
        ],
        monkeypatch,
    )
    turn = runtime.run("read note")
    assert turn.status == TurnStatus.COMPLETED.value
    assert "alpha" in (turn.final_text or "")
    events = runtime.history.events(runtime.session.session_id)
    assert any(event.type == "observation" and event.payload["summary"] == "alpha" for event in events)


def test_ask_user_pending_and_resume(tmp_path, monkeypatch):
    runtime = make_runtime(
        tmp_path,
        [
            ModelResponse(tool_calls=[ToolCall("c-ask", "ask_user", {"question": "Country?"})]),
            ModelResponse(text="France it is"),
        ],
        monkeypatch,
    )
    turn = runtime.run("pick a country")
    assert turn.status == TurnStatus.PENDING.value
    assert turn.wait_ids == ["c-ask"]
    resumed = runtime.resume(runtime.session.session_id, "France")
    assert resumed.status == TurnStatus.COMPLETED.value
    assert resumed.final_text == "France it is"


def test_remember_writes_memory_and_reminder(tmp_path, monkeypatch):
    runtime = make_runtime(
        tmp_path,
        [
            ModelResponse(tool_calls=[ToolCall("c-mem", "remember", {"slot": "work.city", "text": "Hangzhou"})]),
            ModelResponse(text="noted"),
        ],
        monkeypatch,
    )
    runtime.run("I work in Hangzhou")
    records = runtime.memory.active()
    assert records[0].slot == "work.city"
    assert records[0].text == "Hangzhou"
    assert any(event.type == "reminder" for event in runtime.history.events(runtime.session.session_id))


def test_context_uses_checkpoint(tmp_path, monkeypatch):
    db = FakeDB()
    config = make_config(tmp_path)
    monkeypatch.setattr(Config, "db", lambda self: db)
    history = History(config)
    history.append(Event(type="user_message", session_id="s1", payload={"text": "old"}))
    history.append(Event(type="summary_checkpoint", session_id="s1", payload={"summary": "keep this"}))
    history.append(Event(type="user_message", session_id="s1", payload={"text": "new"}))
    builder = ContextBuilder(config, history, Memory(config, "u", "w"))
    messages = builder.messages("s1")
    texts = [item.get("content") for item in messages]
    assert any("keep this" in (text or "") for text in texts)
    assert not any(text and "old" in text and USER_TAG in text for text in texts)
    assert any(text and "new" in text for text in texts)


def test_write_file_tool(tmp_path):
    tools = default_tools(Config(root=tmp_path), tmp_path)
    tools["write_file"].run(path="out.md", content="hi")
    assert (tmp_path / "out.md").read_text(encoding="utf-8") == "hi"


def test_spawn_child(tmp_path, monkeypatch):
    parent = make_runtime(
        tmp_path,
        [
            ModelResponse(tool_calls=[ToolCall("c-spawn", "spawn", {"template": "explore", "goal": "look around"})]),
            ModelResponse(text="child said found it"),
        ],
        monkeypatch,
    )
    child = make_runtime(tmp_path, [ModelResponse(text="found it")], monkeypatch)
    monkeypatch.setattr(Runtime, "create", classmethod(lambda cls, *args, **kwargs: child))
    turn = parent.run("explore")
    assert turn.status == TurnStatus.COMPLETED.value
    events = parent.history.events(parent.session.session_id)
    obs = next(event for event in events if event.type == "observation" and event.payload.get("tool_name") == "spawn")
    assert obs.payload["outcome"] == "pass"
    assert obs.payload["summary"] == "found it"
    assert obs.payload["child_agent_id"] == child.session.agent_id


def test_cancel_flag_and_idempotent_write(tmp_path, monkeypatch):
    store = MemoryStore()
    runtime = make_runtime(
        tmp_path,
        [
            ModelResponse(tool_calls=[ToolCall("c1", "write_file", {"path": "out.md", "content": "once"})]),
            ModelResponse(tool_calls=[ToolCall("c2", "write_file", {"path": "out.md", "content": "once"})]),
            ModelResponse(text="wrote"),
        ],
        monkeypatch,
    )
    runtime.store = store
    turn = runtime.run("write")
    assert turn.status == TurnStatus.COMPLETED.value
    assert (tmp_path / "out.md").read_text(encoding="utf-8") == "once"
    events = [event for event in runtime.history.events(runtime.session.session_id) if event.type == "observation"]
    summaries = [event.payload.get("summary", "") for event in events]
    assert len(summaries) == 2
    assert summaries[0] == summaries[1]

    request_cancel(store, "turn_demo")
    assert is_cancelled(store, "turn_demo")

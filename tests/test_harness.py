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
    """In-memory stand-in for harness.db.Supabase (tables + storage)."""

    def __init__(self):
        self.tables: dict[str, list[dict]] = {}
        self.files: dict[str, str] = {}  # "bucket/path" -> content

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

    def storage_put(self, bucket: str, path: str, content: str) -> None:
        self.files[f"{bucket}/{path}"] = content

    def storage_get(self, bucket: str, path: str) -> str:
        key = f"{bucket}/{path}"
        if key not in self.files:
            raise RuntimeError(f"not found: {path}")
        return self.files[key]

    def storage_list(self, bucket: str, prefix: str = "") -> list[str]:
        prefix_full = f"{bucket}/{prefix}"
        return [k.split("/", 1)[1] for k in self.files if k.startswith(prefix_full)]

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


def make_runtime(tmp_path: Path, responses: list, monkeypatch: pytest.MonkeyPatch, db: FakeDB | None = None) -> Runtime:
    db = db or FakeDB()
    config = make_config(tmp_path)
    monkeypatch.setattr(Config, "db", lambda self: db)
    runtime = Runtime.create(config, cwd=tmp_path, store=MemoryStore())
    runtime.llm = FakeLLM(responses)
    runtime.db = db  # type: ignore[attr-defined]
    return runtime


def test_direct_answer(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path, [ModelResponse(text="hello")], monkeypatch)
    turn = runtime.run("hi")
    assert turn.status == TurnStatus.COMPLETED.value
    assert turn.final_text == "hello"
    assert runtime.history.load_session(runtime.session.session_id) is not None


def test_read_file_tool(tmp_path, monkeypatch):
    db = FakeDB()
    db.storage_put("agent-files", "local/files/note.txt", "alpha")
    runtime = make_runtime(
        tmp_path,
        [
            ModelResponse(tool_calls=[ToolCall("c1", "read_file", {"path": "note.txt"})]),
            ModelResponse(text="the file says alpha"),
        ],
        monkeypatch,
        db=db,
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


def test_should_compact_ignores_pre_checkpoint_events(tmp_path, monkeypatch):
    db = FakeDB()
    config = make_config(tmp_path)
    monkeypatch.setattr(Config, "db", lambda self: db)

    # checkpoint 之前有超大旧事件，checkpoint 之后只有少量新事件 → 不应再压缩
    history = History(config)
    history.append(Event(type="user_message", session_id="s1", payload={"text": "x" * (config.compact_chars + 1000)}))
    history.append(Event(type="summary_checkpoint", session_id="s1", payload={"summary": "small"}))
    history.append(Event(type="user_message", session_id="s1", payload={"text": "new"}))
    builder = ContextBuilder(config, history, Memory(config, "u", "w"))
    assert builder.should_compact("s1") is False

    # checkpoint 之后的新事件又超限 → 需要再次压缩
    history2 = History(config)
    history2.append(Event(type="summary_checkpoint", session_id="s2", payload={"summary": "small"}))
    history2.append(Event(type="user_message", session_id="s2", payload={"text": "y" * (config.compact_chars + 1000)}))
    builder2 = ContextBuilder(config, history2, Memory(config, "u", "w"))
    assert builder2.should_compact("s2") is True


def test_write_file_tool(tmp_path, monkeypatch):
    db = FakeDB()
    config = make_config(tmp_path)
    monkeypatch.setattr(Config, "db", lambda self: db)
    tools = default_tools(config, tmp_path)
    tools["write_file"].run(path="out.md", content="hi")
    assert db.files["agent-files/local/files/out.md"] == "hi"


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
    db = FakeDB()
    runtime = make_runtime(
        tmp_path,
        [
            ModelResponse(tool_calls=[ToolCall("c1", "write_file", {"path": "out.md", "content": "once"}, human_params={"approve": True})]),
            ModelResponse(tool_calls=[ToolCall("c2", "write_file", {"path": "out.md", "content": "once"}, human_params={"approve": True})]),
            ModelResponse(text="wrote"),
        ],
        monkeypatch,
        db=db,
    )
    runtime.store = store
    turn = runtime.run("write")
    assert turn.status == TurnStatus.COMPLETED.value
    assert db.files["agent-files/local/files/out.md"] == "once"
    events = [event for event in runtime.history.events(runtime.session.session_id) if event.type == "observation"]
    summaries = [event.payload.get("summary", "") for event in events]
    assert len(summaries) == 2
    assert summaries[0] == summaries[1]

    request_cancel(store, "turn_demo")
    assert is_cancelled(store, "turn_demo")


def test_cancel_keeps_flag_and_records_history(tmp_path, monkeypatch):
    store = MemoryStore()
    db = FakeDB()
    runtime = make_runtime(tmp_path, [ModelResponse(text="done")], monkeypatch, db=db)
    runtime.store = store
    runtime.run("hi")
    turn = runtime.history.load_turn(runtime.session.session_id)
    runtime.cancel(turn.turn_id)
    # 取消标记仍在（cancel 是 B 侧请求，不 cleanup 删除它）
    assert is_cancelled(store, turn.turn_id)
    # History 里记录了 cancelled
    statuses = [e for e in runtime.history.events(runtime.session.session_id) if e.type == "turn_status"]
    assert any(e.payload.get("status") == TurnStatus.CANCELLED.value for e in statuses)


def test_before_tool_checkpoint_blocks_tools_when_cancelled(tmp_path, monkeypatch):
    db = FakeDB()
    runtime = make_runtime(tmp_path, [], monkeypatch, db=db)
    turn = runtime.start_turn("write two files")
    runtime.cancelled = True
    result = runtime._handle_tools(
        turn,
        [
            ToolCall("c1", "write_file", {"path": "a.md", "content": "1"}),
            ToolCall("c2", "write_file", {"path": "b.md", "content": "2"}),
        ],
    )
    assert result is not None
    assert result.status == TurnStatus.CANCELLED.value
    # before_tool 检查点拦住了所有 tool，没有写入任何文件
    assert "agent-files/local/files/a.md" not in db.files
    assert "agent-files/local/files/b.md" not in db.files


def test_write_file_requires_approval_then_resume(tmp_path, monkeypatch):
    db = FakeDB()
    runtime = make_runtime(
        tmp_path,
        [
            ModelResponse(tool_calls=[ToolCall("c-write", "write_file", {"path": "out.md", "content": "secret"})]),
            ModelResponse(text="wrote after approval"),
        ],
        monkeypatch,
        db=db,
    )
    turn = runtime.run("write a file")
    assert turn.status == TurnStatus.PENDING.value
    assert turn.waiting_for == "human"
    assert "agent-files/local/files/out.md" not in db.files
    resumed = runtime.resume(runtime.session.session_id, "yes")
    assert resumed.status == TurnStatus.COMPLETED.value
    assert db.files["agent-files/local/files/out.md"] == "secret"
    assert any(event.type == "permission" for event in runtime.history.events(runtime.session.session_id))


def test_checkpoint_records_covers_events(tmp_path, monkeypatch):
    db = FakeDB()
    config = make_config(tmp_path)
    monkeypatch.setattr(Config, "db", lambda self: db)
    history = History(config)
    first = history.append(Event(type="user_message", session_id="s3", payload={"text": "old"}))
    builder = ContextBuilder(config, history, Memory(config, "u", "w"))
    builder.compact("s3", "t1", "a1", "keep this", covers_events=builder.covered_event_ids("s3"))
    events = history.events("s3")
    checkpoint = next(event for event in events if event.type == "summary_checkpoint")
    assert checkpoint.payload["covers_events"] == [first.event_id]


def test_read_artifact_range(tmp_path, monkeypatch):
    db = FakeDB()
    config = make_config(tmp_path)
    monkeypatch.setattr(Config, "db", lambda self: db)
    db.insert("artifacts", {"artifact_id": "art_1", "content": "abcdefghij", "preview": "abc"})
    tools = default_tools(config, tmp_path)
    observation = tools["read_artifact_range"].run(artifact_id="art_1", start_byte=2, end_byte=5)
    assert observation.summary == "cdef"
    assert observation.artifact_id == "art_1"


def test_session_start_seeds_balance_and_turn_debits(tmp_path, monkeypatch):
    db = FakeDB()
    runtime = make_runtime(tmp_path, [ModelResponse(text="ok")], monkeypatch, db=db)
    rows = db.select("user_balances", {"user_id": "eq.local"})
    assert rows
    assert int(rows[0]["credits"]) == 1_000_000
    runtime.run("hi")
    rows = db.select("user_balances", {"user_id": "eq.local"})
    assert int(rows[0]["credits"]) == 999_999


def test_insufficient_balance_rejects_turn(tmp_path, monkeypatch):
    db = FakeDB()
    runtime = make_runtime(tmp_path, [ModelResponse(text="should not run")], monkeypatch, db=db)
    db.upsert("user_balances", {"user_id": "local", "credits": 0, "updated_at": "now"}, "user_id")
    turn = runtime.run("hi")
    assert turn.status == TurnStatus.FAILED.value
    assert "insufficient_balance" in (turn.final_text or "")


def test_every_tool_has_hook_lists_and_ask_user_interrupts(tmp_path, monkeypatch):
    db = FakeDB()
    config = make_config(tmp_path)
    monkeypatch.setattr(Config, "db", lambda self: db)
    tools = default_tools(config, tmp_path)
    for tool in tools.values():
        assert isinstance(tool.before_hooks, list)
        assert isinstance(tool.after_hooks, list)
    assert any(hook.name == "interrupt" for hook in tools["ask_user"].before_hooks)
    assert any(hook.name == "approval" for hook in tools["write_file"].before_hooks)
    assert any(hook.name == "approval" for hook in tools["bash"].before_hooks)

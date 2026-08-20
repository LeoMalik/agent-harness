from __future__ import annotations

from pathlib import Path
from threading import Event as ThreadEvent

import pytest

from harness.config import Config
from harness.context import ContextBuilder, USER_TAG
from harness.hooks import HookContext, _session_title_hook, sanitize_session_title
from harness.history import History
from harness.memory import Memory
from harness.runtime import Runtime
from harness.store import MemoryStore, is_cancelled, request_cancel
from harness.tools import default_tools
from harness.user_settings import UserSettingsStore
from harness.types import Event, ModelResponse, ToolCall, Turn, TurnStatus


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
        small_api_key="",
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


def test_skill_catalog_prefers_frontmatter_description(tmp_path, monkeypatch):
    db = FakeDB()
    config = make_config(tmp_path)
    monkeypatch.setattr(Config, "db", lambda self: db)
    (tmp_path / "skills").mkdir(exist_ok=True)
    (tmp_path / "skills" / "find-skills").mkdir(exist_ok=True)
    (tmp_path / "skills" / "find-skills" / "SKILL.md").write_text(
        "---\nname: find-skills\ndescription: Discover and install agent skills.\n---\n\n# Find Skills\n",
        encoding="utf-8",
    )
    history = History(config)
    builder = ContextBuilder(config, history, Memory(config, "u", "w"))
    catalog = builder._skill_catalog()
    assert "find-skills (built-in): Discover and install agent skills." in catalog
    assert "---" not in catalog


def test_tool_call_message_keeps_reasoning_content(tmp_path, monkeypatch):
    db = FakeDB()
    config = make_config(tmp_path)
    monkeypatch.setattr(Config, "db", lambda self: db)
    history = History(config)
    history.append(Event(type="user_message", session_id="s1", payload={"text": "hi"}))
    history.append(
        Event(
            type="tool_call",
            session_id="s1",
            payload={
                "tool_call_id": "call_1",
                "name": "search_web",
                "arguments": {"query": "x"},
                "thinking": "step by step",
            },
        )
    )
    history.append(
        Event(
            type="observation",
            session_id="s1",
            payload={"tool_call_id": "call_1", "tool_name": "search_web", "outcome": "pass", "summary": "ok"},
        )
    )
    builder = ContextBuilder(config, history, Memory(config, "u", "w"))
    messages = builder.messages("s1")
    assistant = next(item for item in messages if item.get("tool_calls"))
    assert assistant["reasoning_content"] == "step by step"
    assert assistant["tool_calls"][0]["function"]["name"] == "search_web"


def test_run_marks_turn_failed_on_exception(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path, [ModelResponse(text="ok")], monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(runtime.llm, "complete", boom)
    turn = runtime.run("hi")
    assert turn.status == TurnStatus.FAILED.value
    assert "RuntimeError" in (turn.final_text or "")
    assert "boom" in (turn.final_text or "")


def test_handle_tools_persists_thinking(tmp_path, monkeypatch):
    runtime = make_runtime(
        tmp_path,
        [
            ModelResponse(thinking="think step", tool_calls=[ToolCall(tool_call_id="call_1", name="search_web", arguments={"query": "x"})]),
            ModelResponse(text="final"),
        ],
        monkeypatch,
    )
    turn = runtime.run("hi")
    tool_calls = [e for e in runtime.history.events(turn.session_id) if e.type == "tool_call"]
    assert tool_calls
    assert tool_calls[0].payload.get("thinking") == "think step"


def test_parse_json_array_handles_fences_and_prose():
    from harness import hooks as hooks_module

    assert hooks_module._parse_json_array('```json\n[{"slot":"a","text":"b"}]\n```') == [{"slot": "a", "text": "b"}]
    assert hooks_module._parse_json_array('prose [{"slot":"a","text":"b"}] tail') == [{"slot": "a", "text": "b"}]
    assert hooks_module._parse_json_array("no array here") == []
    assert hooks_module._parse_json_array('[1, 2, "x"]') == []


def test_memory_replace_active_retires_and_updates(tmp_path, monkeypatch):
    db = FakeDB()
    config = make_config(tmp_path)
    monkeypatch.setattr(Config, "db", lambda self: db)
    memory = Memory(config, "u", "w")
    memory.upsert("profile.a", "old-a", "profile", "t1", "s1")
    memory.upsert("profile.b", "old-b", "profile", "t1", "s1")
    memory.replace_active([{"slot": "profile.a", "text": "new-a", "layer": "profile"}], "t2", "s2")
    assert {f.slot: f.text for f in memory.active()} == {"profile.a": "new-a"}
    assert any(f.slot == "profile.b" and f.status == "retired" for f in memory.load())


def test_persist_memory_extracts_facts(tmp_path, monkeypatch):
    from dataclasses import replace
    from harness import hooks as hooks_module

    runtime = make_runtime(tmp_path, [ModelResponse(text="x")], monkeypatch)
    runtime.config = replace(runtime.config, small_api_key="test-key")

    class FakeMemoryLLM:
        def __init__(self, *args, **kwargs):
            pass

        def complete(self, messages, tools=None, temperature=0.2, on_delta=None):
            return ModelResponse(text='[{"slot":"profile.location.city","text":"杭州","layer":"profile"}]')

    monkeypatch.setattr(hooks_module, "LLM", FakeMemoryLLM)
    turn = Turn(
        turn_id="turn_mem",
        session_id=runtime.session.session_id,
        agent_id="a",
        user_text="我住在杭州",
        final_text="好的",
    )
    hooks_module._persist_memory(HookContext(event="after_turn", runtime=runtime, turn=turn))
    facts = runtime.memory.active()
    assert any(f.slot == "profile.location.city" and f.text == "杭州" for f in facts)


def test_reflect_memory_fires_every_n_turns(tmp_path, monkeypatch):
    from dataclasses import replace
    from harness import hooks as hooks_module

    runtime = make_runtime(tmp_path, [ModelResponse(text="x")], monkeypatch)
    runtime.config = replace(runtime.config, reflect_interval=2, small_api_key="test-key")

    class FakeReflectLLM:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def complete(self, messages, tools=None, temperature=0.2, on_delta=None):
            self.calls += 1
            return ModelResponse(text='[{"slot":"profile.a","text":"merged","layer":"profile"}]')

    fake = FakeReflectLLM()
    monkeypatch.setattr(hooks_module, "LLM", lambda *a, **k: fake)
    runtime.memory.upsert("profile.a", "old", "profile", "t0", "s0")
    turn = Turn(turn_id="turn_ref", session_id=runtime.session.session_id, agent_id="a")
    ctx = HookContext(event="after_turn", runtime=runtime, turn=turn)

    hooks_module._reflect_memory(ctx)  # count = 1
    assert fake.calls == 0
    hooks_module._reflect_memory(ctx)  # count = 2 → reflect
    assert fake.calls == 1
    assert any(f.slot == "profile.a" and f.text == "merged" for f in runtime.memory.active())


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


def test_user_settings_are_separate_and_feed_runtime(tmp_path, monkeypatch):
    db = FakeDB()
    config = make_config(tmp_path)
    monkeypatch.setattr(Config, "db", lambda self: db)
    store = UserSettingsStore(config)
    saved = store.update(
        "u1",
        {
            "default_model": "custom/model",
            "reasoning_effort": "high",
            "soul_md": "# SOUL.md\nAlways say lighthouse.",
            "settings": {"theme": "light"},
        },
    )
    assert saved.reasoning_effort == "high"
    runtime = Runtime.create(config, user_id="u1", workspace_id="studio", cwd=tmp_path, store=MemoryStore())
    assert runtime.llm.model == "custom/model"
    assert runtime.llm.reasoning_effort == "high"
    prompt = runtime.context.system_prompt(soul_override=runtime.user_settings.soul_md)
    assert "Always say lighthouse" in prompt
    assert "model" not in runtime.session.to_dict()
    assert "soul_md" not in runtime.session.to_dict()


def test_session_metadata_lists_workspaces_and_filters(tmp_path, monkeypatch):
    db = FakeDB()
    config = make_config(tmp_path)
    monkeypatch.setattr(Config, "db", lambda self: db)
    history = History(config)
    runtime = Runtime.create(config, user_id="u2", workspace_id="art", cwd=tmp_path, store=MemoryStore())
    runtime.session.title = "Concept sketch"
    runtime.session.starred = True
    history.save_session(runtime.session)
    listed = history.list_sessions("u2", "art")
    assert listed[0].title == "Concept sketch"
    assert listed[0].starred is True
    assert history.list_workspaces("u2") == ["art"]


def test_llm_payload_includes_reasoning_effort(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    from harness.llm import LLM

    llm = LLM(config, model_override="custom/model", reasoning_effort="high")
    captured = {}
    monkeypatch.setattr(llm, "_stream", lambda body, on_delta: captured.update(body) or ModelResponse(text="ok"))
    monkeypatch.setattr(type(llm), "api_key", property(lambda self: "key"))
    llm.complete([{"role": "user", "content": "hi"}])
    assert captured["model"] == "custom/model"
    assert captured["reasoning_effort"] == "high"


def test_llm_falls_back_before_first_stream_delta(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    from harness.llm import LLM

    llm = LLM(config)
    calls = []
    monkeypatch.setattr(type(llm), "api_key", property(lambda self: "key"))
    monkeypatch.setattr(llm, "_stream", lambda body, on_delta: (_ for _ in ()).throw(ConnectionError("no stream")))
    monkeypatch.setattr(
        llm,
        "_once",
        lambda body, on_delta: calls.append(dict(body)) or ModelResponse(text="fallback"),
    )

    response = llm.complete([{"role": "user", "content": "hi"}])

    assert response.text == "fallback"
    assert calls == [{"model": llm.model, "messages": [{"role": "user", "content": "hi"}], "temperature": 0.2, "stream": False, "reasoning_effort": "medium"}]


def test_llm_does_not_fallback_after_stream_delta(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    from harness.llm import LLM

    llm = LLM(config)
    deltas = []
    fallback_called = False
    monkeypatch.setattr(type(llm), "api_key", property(lambda self: "key"))

    def partial_stream(body, on_delta):
        on_delta("thinking", "partial")
        raise ConnectionError("stream interrupted")

    def fallback(body, on_delta):
        nonlocal fallback_called
        fallback_called = True
        return ModelResponse(text="duplicate")

    monkeypatch.setattr(llm, "_stream", partial_stream)
    monkeypatch.setattr(llm, "_once", fallback)

    with pytest.raises(ConnectionError, match="stream interrupted"):
        llm.complete([{"role": "user", "content": "hi"}], on_delta=lambda kind, text: deltas.append((kind, text)))

    assert deltas == [("thinking", "partial")]
    assert fallback_called is False


def test_title_sanitizer_enforces_concise_contract():
    chinese = sanitize_session_title('标题：**讨论新的绘画工作区设计方案。**', '请设计一个新的绘画工作区')
    assert chinese == '讨论新的绘画工作区设计方'
    assert len(chinese) <= 12
    english = sanitize_session_title('Title: "Design a better workspace conversation sidebar today."', 'Design a better sidebar')
    assert english == 'Design a better workspace conversation sidebar today'
    assert len(english.split()) <= 8


def test_session_title_generation_does_not_block_main_answer(tmp_path, monkeypatch):
    db = FakeDB()
    config = Config(
        root=tmp_path,
        prompts_dir=tmp_path / "prompts",
        skills_dir=tmp_path / "skills",
        graphs_dir=tmp_path / "graphs",
        small_api_key="test-key",
    )
    (tmp_path / "prompts").mkdir(exist_ok=True)
    (tmp_path / "prompts" / "AGENTS.md").write_text("Follow the user.", encoding="utf-8")
    (tmp_path / "prompts" / "SOUL.md").write_text("Be brief.", encoding="utf-8")
    monkeypatch.setattr(Config, "db", lambda self: db)
    from harness import hooks as hooks_module

    title_started = ThreadEvent()
    release_title = ThreadEvent()

    class BlockingTitleLLM:
        model = "small-title-model"

        def __init__(self, *args, **kwargs):
            pass

        def complete(self, *args, **kwargs):
            title_started.set()
            release_title.wait(timeout=2)
            return ModelResponse(text="Async title")

    monkeypatch.setattr(hooks_module, "LLM", BlockingTitleLLM)
    monkeypatch.setattr(hooks_module, "_persist_memory", lambda ctx: None)
    monkeypatch.setattr(hooks_module, "_reflect_memory", lambda ctx: None)
    runtime = Runtime.create(config, user_id="u-async", cwd=tmp_path, store=MemoryStore())
    runtime.llm = FakeLLM([ModelResponse(text="main answer")])
    turn = runtime.run("Please answer while naming this chat")
    assert title_started.wait(timeout=1)
    assert turn.status == TurnStatus.COMPLETED.value
    assert turn.final_text == "main answer"
    assert runtime.history.load_session(runtime.session.session_id).title == ""
    release_title.set()
    for _ in range(100):
        if runtime.history.load_session(runtime.session.session_id).title:
            break
        ThreadEvent().wait(0.01)
    assert runtime.history.load_session(runtime.session.session_id).title == "Async title"


def test_session_title_hook_updates_only_first_untitled_session(tmp_path, monkeypatch):
    from dataclasses import replace

    db = FakeDB()
    config = replace(make_config(tmp_path), small_api_key="test-key")
    monkeypatch.setattr(Config, "db", lambda self: db)
    runtime = Runtime.create(config, user_id="u-title", cwd=tmp_path, store=MemoryStore())
    turn = runtime.start_turn("请帮我规划绘画工作区")
    from harness import hooks as hooks_module

    class TitleLLM:
        model = "small-title-model"

        def __init__(self, *args, **kwargs):
            pass

        def complete(self, *args, **kwargs):
            return ModelResponse(text="绘画工作区规划。")

    monkeypatch.setattr(hooks_module, "LLM", TitleLLM)
    _session_title_hook(HookContext("user_prompt_submit", runtime, turn, extra={"user_text": turn.user_text}))
    saved = runtime.history.load_session(runtime.session.session_id)
    assert saved.title == "绘画工作区规划"
    events = runtime.store.xrange(f"turn:{turn.turn_id}:events")
    assert any(fields.get("type") == "session.title_updated" for _, fields in events)
    _session_title_hook(HookContext("user_prompt_submit", runtime, turn, extra={"user_text": "第二条"}))
    assert runtime.history.load_session(runtime.session.session_id).title == "绘画工作区规划"


def test_session_title_failure_does_not_fail_turn(tmp_path, monkeypatch):
    db = FakeDB()
    runtime = make_runtime(tmp_path, [ModelResponse(text="main answer")], monkeypatch, db=db)
    from harness import hooks as hooks_module

    class BrokenTitleLLM:
        def __init__(self, *args, **kwargs):
            pass

        def complete(self, *args, **kwargs):
            raise RuntimeError("title unavailable")

    monkeypatch.setattr(hooks_module, "LLM", BrokenTitleLLM)
    runtime.config = Config(
        root=runtime.config.root,
        prompts_dir=runtime.config.prompts_dir,
        skills_dir=runtime.config.skills_dir,
        graphs_dir=runtime.config.graphs_dir,
        small_api_key="test-key",
    )
    turn = runtime.run("answer normally")
    assert turn.status == TurnStatus.COMPLETED.value
    assert turn.final_text == "main answer"

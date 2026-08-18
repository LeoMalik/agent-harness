from __future__ import annotations

from pathlib import Path

from harness.config import Config
from harness.context import ContextBuilder, USER_TAG
from harness.history import History
from harness.memory import Memory
from harness.runtime import Runtime
from harness.store import MemoryStore, is_cancelled, request_cancel
from harness.tools import default_tools
from harness.types import Event, ModelResponse, ToolCall, TurnStatus


class FakeLLM:
    def __init__(self, responses: list[ModelResponse]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def complete(self, messages, tools=None, temperature=0.2):
        self.calls.append(messages)
        if not self.responses:
            return ModelResponse(text="empty")
        return self.responses.pop(0)


def make_runtime(tmp_path: Path, responses: list[ModelResponse]) -> Runtime:
    config = Config(root=tmp_path, data_dir=tmp_path / "data", prompts_dir=tmp_path / "prompts", skills_dir=tmp_path / "skills", graphs_dir=tmp_path / "graphs")
    (tmp_path / "prompts").mkdir(exist_ok=True)
    (tmp_path / "prompts" / "AGENTS.md").write_text("Follow the user.", encoding="utf-8")
    (tmp_path / "prompts" / "SOUL.md").write_text("Be brief.", encoding="utf-8")
    runtime = Runtime.create(config, cwd=tmp_path, store=MemoryStore())
    runtime.llm = FakeLLM(responses)
    return runtime


def test_direct_answer(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path, [ModelResponse(text="hello")])
    turn = runtime.run("hi")
    assert turn.status == TurnStatus.COMPLETED.value
    assert turn.final_text == "hello"
    assert runtime.history.load_session(runtime.session.session_id) is not None


def test_read_file_tool(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("alpha", encoding="utf-8")
    runtime = make_runtime(
        tmp_path,
        [
            ModelResponse(
                tool_calls=[ToolCall("c1", "read_file", {"path": "note.txt"})],
            ),
            ModelResponse(text="the file says alpha"),
        ],
    )
    turn = runtime.run("read note")
    assert turn.status == TurnStatus.COMPLETED.value
    assert "alpha" in (turn.final_text or "")
    events = runtime.history.events(runtime.session.session_id)
    assert any(event.type == "observation" and event.payload["summary"] == "alpha" for event in events)


def test_ask_user_pending_and_resume(tmp_path: Path) -> None:
    runtime = make_runtime(
        tmp_path,
        [
            ModelResponse(tool_calls=[ToolCall("c-ask", "ask_user", {"question": "Country?"})]),
            ModelResponse(text="France it is"),
        ],
    )
    turn = runtime.run("pick a country")
    assert turn.status == TurnStatus.PENDING.value
    assert turn.wait_ids == ["c-ask"]
    resumed = runtime.resume(runtime.session.session_id, "France")
    assert resumed.status == TurnStatus.COMPLETED.value
    assert resumed.final_text == "France it is"


def test_remember_writes_memory_and_reminder(tmp_path: Path) -> None:
    runtime = make_runtime(
        tmp_path,
        [
            ModelResponse(tool_calls=[ToolCall("c-mem", "remember", {"slot": "work.city", "text": "Hangzhou"})]),
            ModelResponse(text="noted"),
        ],
    )
    runtime.run("I work in Hangzhou")
    records = runtime.memory.active()
    assert records[0].slot == "work.city"
    assert records[0].text == "Hangzhou"
    assert any(event.type == "reminder" for event in runtime.history.events(runtime.session.session_id))


def test_context_uses_checkpoint(tmp_path: Path) -> None:
    config = Config(root=tmp_path, data_dir=tmp_path / "data", prompts_dir=tmp_path / "prompts")
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "AGENTS.md").write_text("sys", encoding="utf-8")
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


def test_write_file_tool(tmp_path: Path) -> None:
    tools = default_tools(Config(data_dir=tmp_path / "data"), tmp_path)
    tools["write_file"].run(path="out.md", content="hi")
    assert (tmp_path / "out.md").read_text(encoding="utf-8") == "hi"


def test_spawn_child(tmp_path: Path, monkeypatch) -> None:
    parent = make_runtime(
        tmp_path,
        [
            ModelResponse(tool_calls=[ToolCall("c-spawn", "spawn", {"template": "explore", "goal": "look around"})]),
            ModelResponse(text="child said found it"),
        ],
    )
    child = make_runtime(tmp_path, [ModelResponse(text="found it")])
    monkeypatch.setattr(Runtime, "create", classmethod(lambda cls, *args, **kwargs: child))
    turn = parent.run("explore")
    assert turn.status == TurnStatus.COMPLETED.value
    events = parent.history.events(parent.session.session_id)
    obs = next(event for event in events if event.type == "observation" and event.payload.get("tool_name") == "spawn")
    assert obs.payload["outcome"] == "pass"
    assert obs.payload["summary"] == "found it"
    assert obs.payload["child_agent_id"] == child.session.agent_id


def test_cancel_flag_and_idempotent_write(tmp_path: Path) -> None:
    store = MemoryStore()
    runtime = make_runtime(
        tmp_path,
        [
            ModelResponse(tool_calls=[ToolCall("c1", "write_file", {"path": "out.md", "content": "once"})]),
            ModelResponse(tool_calls=[ToolCall("c2", "write_file", {"path": "out.md", "content": "once"})]),
            ModelResponse(text="wrote"),
        ],
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

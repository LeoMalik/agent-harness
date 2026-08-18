from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.config import Config
from harness.context import ContextBuilder
from harness.history import History
from harness.llm import LLM
from harness.memory import Memory
from harness.store import (
    Store,
    cleanup_turn,
    connect_store,
    is_cancelled,
    publish_event,
    request_cancel,
    resources_key,
)
from harness.tools import Tool, bind, default_tools, spawn_schema
from harness.types import (
    Observation,
    Session,
    ToolCall,
    Turn,
    TurnStatus,
    WaitingFor,
    new_id,
)


TEMPLATES = {
    "general": "Finish the assigned goal. Stay inside the goal. Return a short Observation.",
    "explore": "Read-only. Do not write files or run mutating shell commands. Return paths and conclusions.",
    "test": "Run tests, read failures, report what failed. Do not change production code unless asked.",
}


@dataclass
class Runtime:
    config: Config
    history: History
    memory: Memory
    llm: LLM
    context: ContextBuilder
    cwd: Path
    session: Session
    tools: dict[str, Tool]
    store: Store
    cancelled: bool = False
    depth: int = 0
    extra_system: str = ""
    allowed_tools: set[str] | None = None

    @classmethod
    def create(
        cls,
        config: Config | None = None,
        *,
        user_id: str = "local",
        workspace_id: str = "default",
        cwd: Path | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        parent_agent_id: str | None = None,
        depth: int = 0,
        extra_system: str = "",
        allowed_tools: set[str] | None = None,
        store: Store | None = None,
    ) -> Runtime:
        config = config or Config()
        cwd = (cwd or Path.cwd()).resolve()
        history = History(config)
        memory = Memory(config, user_id, workspace_id)
        store = store or connect_store(config.redis_url)
        if session_id and (existing := history.load_session(session_id)):
            session = existing
        else:
            session = Session(
                session_id=session_id or new_id("ses"),
                agent_id=new_id("agt"),
                user_id=user_id,
                workspace_id=workspace_id,
                parent_session_id=parent_session_id,
                parent_agent_id=parent_agent_id,
            )
            history.save_session(session)
        return cls(
            config=config,
            history=history,
            memory=memory,
            llm=LLM(config, small=depth > 0),
            context=ContextBuilder(config, history, memory),
            cwd=cwd,
            session=session,
            tools=default_tools(config, cwd),
            store=store,
            depth=depth,
            extra_system=extra_system,
            allowed_tools=allowed_tools,
        )

    def schemas(self) -> list[dict[str, Any]]:
        names = self.allowed_tools or set(self.tools)
        items = [self.tools[name].schema() for name in sorted(names) if name in self.tools]
        if self.depth < self.config.max_agent_depth and (self.allowed_tools is None or "spawn" in self.allowed_tools):
            items.append(spawn_schema())
        return items

    def start_turn(self, text: str) -> Turn:
        turn = Turn(
            turn_id=new_id("turn"),
            session_id=self.session.session_id,
            agent_id=self.session.agent_id,
            user_text=text,
        )
        self.history.save_turn(turn)
        self.history.emit(
            "user_message",
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            agent_id=turn.agent_id,
            text=text,
        )
        publish_event(self.store, turn.turn_id, "turn.started", {"session_id": turn.session_id, "text": text})
        return turn

    def run(self, text: str) -> Turn:
        return self._loop(self.start_turn(text))

    def continue_turn(self) -> Turn:
        turn = self.history.load_turn(self.session.session_id)
        if turn is None:
            raise RuntimeError("No turn to continue")
        return self._loop(turn)

    def resume(self, session_id: str, answer: str) -> Turn:
        turn = self.history.load_turn(session_id)
        if not turn or turn.status != TurnStatus.PENDING.value:
            raise RuntimeError("No pending turn to resume")
        if turn.waiting_for != WaitingFor.HUMAN.value:
            raise RuntimeError("Pending turn is waiting for children, not a human answer")
        self.history.emit(
            "human_params",
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            agent_id=turn.agent_id,
            text=answer,
            wait_ids=turn.wait_ids,
        )
        call_id = turn.wait_ids[0] if turn.wait_ids else new_id("call")
        observation = Observation(
            tool_call_id=call_id,
            tool_name="ask_user",
            summary=answer,
        )
        self._write_observation(turn, observation)
        turn.status = TurnStatus.RUNNING.value
        turn.waiting_for = None
        turn.wait_ids = []
        turn.resume_token = None
        return self._loop(turn)

    def cancel(self, turn_id: str | None = None) -> None:
        self.cancelled = True
        if turn_id:
            request_cancel(self.store, turn_id)
            turn = self.history.load_turn(self.session.session_id)
            if turn and turn.turn_id == turn_id:
                self._finish(turn, TurnStatus.CANCELLED)

    def _stop_if_cancelled(self, turn: Turn) -> Turn | None:
        if self.cancelled or is_cancelled(self.store, turn.turn_id):
            return self._finish(turn, TurnStatus.CANCELLED)
        return None

    def _on_llm_delta(self, turn: Turn, kind: str, text: str) -> None:
        if not text:
            return
        event_type = "thinking.delta" if kind == "thinking" else "assistant.delta"
        publish_event(self.store, turn.turn_id, event_type, {"text": text})

    def _loop(self, turn: Turn) -> Turn:
        self.history.save_turn(turn)
        for _ in range(self.config.max_steps):
            if stopped := self._stop_if_cancelled(turn):
                return stopped
            if self.context.should_compact(turn.session_id):
                self._compact(turn)
            messages = self.context.messages(turn.session_id, self.extra_system)
            response = self.llm.complete(
                messages,
                self.schemas(),
                on_delta=lambda kind, text, current=turn: self._on_llm_delta(current, kind, text),
            )
            if response.tool_calls:
                pending = self._handle_tools(turn, response.tool_calls)
                if pending:
                    return pending
                continue
            turn.final_text = response.text.strip()
            self.history.emit(
                "assistant_message",
                session_id=turn.session_id,
                turn_id=turn.turn_id,
                agent_id=turn.agent_id,
                text=turn.final_text,
            )
            return self._finish(turn, TurnStatus.COMPLETED)
        return self._finish(turn, TurnStatus.FAILED, "max_steps_exceeded")

    def _handle_tools(self, turn: Turn, calls: list[ToolCall]) -> Turn | None:
        wait_ids: list[str] = []
        for call in calls:
            self.history.emit(
                "tool_call",
                session_id=turn.session_id,
                turn_id=turn.turn_id,
                agent_id=turn.agent_id,
                tool_call_id=call.tool_call_id,
                name=call.name,
                arguments=call.arguments,
            )
            if call.name == "ask_user":
                turn.status = TurnStatus.PENDING.value
                turn.waiting_for = WaitingFor.HUMAN.value
                turn.wait_ids = [call.tool_call_id]
                turn.resume_token = new_id("tok")
                self.history.save_turn(turn)
                publish_event(
                    self.store,
                    turn.turn_id,
                    "ask_user",
                    {"question": (call.arguments or {}).get("question") or "", "tool_call_id": call.tool_call_id},
                )
                return turn
            if call.name == "spawn":
                wait_ids.append(call.tool_call_id)
                observation = self._spawn(turn, call)
                self._write_observation(turn, observation)
                continue
            if call.name == "remember":
                record = self.memory.upsert(
                    slot=str(call.arguments["slot"]),
                    text=str(call.arguments["text"]),
                    layer=str(call.arguments.get("layer") or "profile"),
                    source_turn_id=turn.turn_id,
                    source_session_id=turn.session_id,
                )
                self.history.emit(
                    "reminder",
                    session_id=turn.session_id,
                    turn_id=turn.turn_id,
                    agent_id=turn.agent_id,
                    text=f"Memory upserted: {record.slot} = {record.text}",
                )
                self._write_observation(
                    turn,
                    bind(
                        Observation(
                            tool_call_id=call.tool_call_id,
                            tool_name="remember",
                            summary=f"Stored {record.slot}",
                            refs=[record.slot],
                        ),
                        call,
                    ),
                )
                continue
            tool = self.tools.get(call.name)
            if tool is None:
                self._write_observation(
                    turn,
                    Observation(
                        tool_call_id=call.tool_call_id,
                        tool_name=call.name,
                        outcome="fail",
                        summary=f"Unknown tool {call.name}",
                        error="unknown_tool",
                    ),
                )
                continue
            cached = self._begin_write(turn, call)
            if cached is not None:
                self._write_observation(turn, cached)
                continue
            publish_event(
                self.store,
                turn.turn_id,
                "tool.started",
                {"tool_call_id": call.tool_call_id, "name": call.name, "arguments": call.arguments},
            )
            try:
                observation = bind(tool.run(**call.arguments), call)
            except Exception as exc:  # noqa: BLE001 - tool errors become observations
                observation = Observation(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.name,
                    outcome="fail",
                    summary=str(exc),
                    error=type(exc).__name__,
                )
            self._complete_write(turn, call, observation)
            self._write_observation(turn, observation)
        if wait_ids:
            # Children run inline in v1; pending is recorded only if a child stays open.
            open_ids = [item for item in wait_ids if self._child_still_open(item)]
            if open_ids:
                turn.status = TurnStatus.PENDING.value
                turn.waiting_for = WaitingFor.CHILDREN.value
                turn.wait_ids = open_ids
                self.history.save_turn(turn)
                return turn
        return None

    def _spawn(self, turn: Turn, call: ToolCall) -> Observation:
        template = str(call.arguments.get("template") or "general")
        goal = str(call.arguments.get("goal") or "")
        graph = call.arguments.get("graph")
        extra = TEMPLATES.get(template, TEMPLATES["general"])
        if graph:
            path = self.config.graphs_dir / f"{graph}.md"
            if path.exists():
                extra += "\n\nFollow this graph:\n" + path.read_text(encoding="utf-8")
        allowed = None
        if template == "explore":
            allowed = {"read_file", "search_web", "ask_user"}
        child = Runtime.create(
            self.config,
            user_id=self.session.user_id,
            workspace_id=self.session.workspace_id,
            cwd=self.cwd,
            parent_session_id=self.session.session_id,
            parent_agent_id=self.session.agent_id,
            depth=self.depth + 1,
            extra_system=extra,
            allowed_tools=allowed,
            store=self.store,
        )
        child_turn = child.run(goal)
        outcome = "pass" if child_turn.status == TurnStatus.COMPLETED.value else "fail"
        if child_turn.status == TurnStatus.PENDING.value:
            outcome = "partial"
        return Observation(
            tool_call_id=call.tool_call_id,
            tool_name="spawn",
            outcome=outcome,
            summary=child_turn.final_text or child_turn.status,
            refs=[],
            child_agent_id=child.session.agent_id,
        )

    def _child_still_open(self, _tool_call_id: str) -> bool:
        return False

    def _write_key(self, turn: Turn, call: ToolCall) -> str:
        payload = json.dumps(
            {"tool": call.name, "arguments": call.arguments, "human": call.human_params},
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
        return f"turn:{turn.turn_id}:tool:idem:{digest}"

    def _begin_write(self, turn: Turn, call: ToolCall) -> Observation | None:
        if call.name not in {"write_file", "bash"}:
            return None
        key = self._write_key(turn, call)
        claimed = self.store.set(key, "running", nx=True, ex=86400)
        self.store.sadd(resources_key(turn.turn_id), key)
        if claimed:
            return None
        raw = self.store.get(key)
        if raw and raw not in {"running"}:
            try:
                return Observation.from_dict(json.loads(raw))
            except json.JSONDecodeError:
                return None
        return Observation(
            tool_call_id=call.tool_call_id,
            tool_name=call.name,
            summary="duplicate write skipped",
            preview="idempotent",
        )

    def _complete_write(self, turn: Turn, call: ToolCall, observation: Observation) -> None:
        if call.name not in {"write_file", "bash"}:
            return
        self.store.set(self._write_key(turn, call), json.dumps(observation.to_dict(), ensure_ascii=False), ex=86400)

    def _write_observation(self, turn: Turn, observation: Observation) -> None:
        self.history.emit(
            "observation",
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            agent_id=turn.agent_id,
            **observation.to_dict(),
        )
        publish_event(self.store, turn.turn_id, "tool.completed", observation.to_dict())

    def _compact(self, turn: Turn) -> None:
        compact_llm = LLM(self.config, small=True)
        messages = [
            {
                "role": "system",
                "content": "Summarize the conversation for resume. Keep the user goal, constraints, decisions, facts, and next step. Do not invent.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    [event.to_dict() for event in self.history.events(turn.session_id)][-40:],
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            summary = compact_llm.complete(messages, tools=None).text.strip()
        except Exception:  # noqa: BLE001
            return  # 静默跳过：不写误导性的 checkpoint
        self.context.compact(turn.session_id, turn.turn_id, turn.agent_id, summary)

    def _finish(self, turn: Turn, status: TurnStatus, error: str | None = None) -> Turn:
        turn.status = status.value
        if error:
            turn.final_text = turn.final_text or error
        self.history.save_turn(turn)
        publish_event(self.store, turn.turn_id, f"turn.{turn.status}", {"error": error})
        if status.value in {TurnStatus.COMPLETED.value, TurnStatus.FAILED.value, TurnStatus.CANCELLED.value}:
            cleanup_turn(self.store, turn.turn_id)
        self.history.emit(
            "turn_status",
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            agent_id=turn.agent_id,
            status=turn.status,
            error=error,
        )
        return turn

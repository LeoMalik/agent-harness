from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.config import Config
from harness.context import ContextBuilder
from harness.executor import ToolExecutor
from harness.history import History
from harness.hooks import (
    AFTER_LLM_CALL,
    AFTER_TOOL_CALL,
    AFTER_TURN,
    BEFORE_LLM_CALL,
    BEFORE_TOOL,
    PRE_COMPACT,
    SESSION_START,
    TURN_CLEANUP,
    USER_PROMPT_SUBMIT,
    HookBus,
    HookContext,
    HookResult,
    default_hooks,
    new_resume_token,
    parse_resume_answer,
)
from harness.llm import LLM
from harness.memory import Memory
from harness.store import (
    Store,
    cleanup_turn,
    connect_store,
    is_cancelled,
    publish_event,
    request_cancel,
)
from harness.tools import Tool, bind, default_tools, spawn_schema
from harness.user_settings import UserSettings, UserSettingsStore
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
    hooks: HookBus
    executor: ToolExecutor
    user_settings: UserSettings
    cancelled: bool = False
    depth: int = 0
    extra_system: str = ""
    allowed_tools: set[str] | None = None
    sub_agent_ids: list[str] = field(default_factory=list)

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
        hooks: HookBus | None = None,
        executor: ToolExecutor | None = None,
    ) -> Runtime:
        config = config or Config()
        cwd = (cwd or Path.cwd()).resolve()
        history = History(config)
        store = store or connect_store(config.redis_url)
        created = False
        if session_id and (existing := history.load_session(session_id)):
            session = existing
        else:
            created = True
            session = Session(
                session_id=session_id or new_id("ses"),
                agent_id=new_id("agt"),
                user_id=user_id,
                workspace_id=workspace_id,
                parent_session_id=parent_session_id,
                parent_agent_id=parent_agent_id,
            )
            history.save_session(session)
        memory = Memory(config, session.user_id, session.workspace_id)
        user_settings = UserSettingsStore(config).get(session.user_id)
        runtime = cls(
            config=config,
            history=history,
            memory=memory,
            llm=LLM(
                config,
                small=depth > 0,
                model_override=None if depth > 0 else user_settings.default_model,
                reasoning_effort=user_settings.reasoning_effort,
            ),
            context=ContextBuilder(config, history, memory),
            cwd=cwd,
            session=session,
            tools=default_tools(config, cwd, session.user_id),
            store=store,
            hooks=hooks or default_hooks(config.hook_timeout_seconds),
            executor=executor or ToolExecutor(),
            user_settings=user_settings,
            depth=depth,
            extra_system=extra_system,
            allowed_tools=allowed_tools,
        )
        if created:
            runtime.on_session_start()
        return runtime

    def publisher(self, turn_id: str, event_type: str, payload: dict[str, Any]) -> str:
        return publish_event(self.store, turn_id, event_type, payload)

    def schemas(self) -> list[dict[str, Any]]:
        names = self.allowed_tools or set(self.tools)
        items = [self.tools[name].schema() for name in sorted(names) if name in self.tools]
        if self.depth < self.config.max_agent_depth and (self.allowed_tools is None or "spawn" in self.allowed_tools):
            items.append(spawn_schema())
        return items

    def run(self, text: str) -> Turn:
        return self._run_loop_safely(self.start_turn(text))

    def _run_loop_safely(self, turn: Turn) -> Turn:
        try:
            return self._loop(turn)
        except Exception as exc:  # noqa: BLE001 - 异常转成失败 turn，避免卡在 running
            return self._finish(turn, TurnStatus.FAILED, f"{type(exc).__name__}: {exc}")

    def continue_turn(self) -> Turn:
        turn = self.history.load_turn(self.session.session_id)
        if turn is None:
            raise RuntimeError("No turn to continue")
        return self._run_loop_safely(turn)

    def on_session_start(self) -> None:
        self.hooks.run_before(SESSION_START, self._hook_ctx(SESSION_START, None))

    def start_turn(self, text: str) -> Turn:
        turn = Turn(
            turn_id=new_id("turn"),
            session_id=self.session.session_id,
            agent_id=self.session.agent_id,
            user_text=text,
        )
        self.history.save_turn(turn)
        self.session.unread = False
        self.history.save_session(self.session)
        result = self.hooks.run_before(USER_PROMPT_SUBMIT, self._hook_ctx(USER_PROMPT_SUBMIT, turn))
        if result.is_rejected:
            turn.status = TurnStatus.FAILED.value
            turn.final_text = result.reason or "rejected"
            self.history.save_turn(turn)
            return turn
        self.history.emit(
            "user_message",
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            agent_id=turn.agent_id,
            text=text,
        )
        self.publisher(turn.turn_id, "turn.started", {"session_id": turn.session_id, "text": text})
        self.hooks.dispatch_async(
            USER_PROMPT_SUBMIT,
            self._hook_ctx(USER_PROMPT_SUBMIT, turn, extra={"user_text": text}),
        )
        return turn

    def resume(self, session_id: str, answer: str) -> Turn:
        turn = self.history.load_turn(session_id)
        if not turn or turn.status != TurnStatus.PENDING.value:
            raise RuntimeError("No pending turn to resume")
        if turn.waiting_for != WaitingFor.HUMAN.value:
            raise RuntimeError("Pending turn is waiting for children, not a human answer")
        call = self._pending_tool_call(turn)
        call.human_params = parse_resume_answer(call.name, answer)
        self.history.emit(
            "human_params",
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            agent_id=turn.agent_id,
            text=answer,
            wait_ids=turn.wait_ids,
            tool_call_id=call.tool_call_id,
            tool_name=call.name,
            human_params=call.human_params,
        )
        turn.status = TurnStatus.RUNNING.value
        turn.waiting_for = None
        turn.wait_ids = []
        turn.resume_token = None
        pending = self._handle_tools(turn, [call], replay=True)
        if pending:
            return pending
        return self._run_loop_safely(turn)

    def cancel(self, turn_id: str | None = None) -> None:
        """外部/B 侧只写共享标记 + History，不 cleanup、不 finish。"""
        self.cancelled = True
        turn = self.history.load_turn(self.session.session_id)
        if turn_id is None:
            turn_id = turn.turn_id if turn else None
        if not turn_id:
            return
        request_cancel(self.store, turn_id)
        self.history.emit(
            "turn_status",
            session_id=turn.session_id if turn else self.session.session_id,
            turn_id=turn_id,
            agent_id=turn.agent_id if turn else self.session.agent_id,
            status=TurnStatus.CANCELLED.value,
            error=None,
        )
        self.publisher(turn_id, "turn.cancelled", {})

    def _hook_ctx(
        self,
        event: str,
        turn: Turn | None,
        *,
        call: ToolCall | None = None,
        tool: Tool | None = None,
        observation: Observation | None = None,
        extra: dict[str, Any] | None = None,
    ) -> HookContext:
        return HookContext(
            event=event,
            runtime=self,
            turn=turn,
            call=call,
            tool=tool,
            observation=observation,
            extra=extra or {},
        )

    def _stop_if_cancelled(self, turn: Turn) -> Turn | None:
        if self.cancelled or is_cancelled(self.store, turn.turn_id):
            self._cancel_agent_tree()
            return self._finish(turn, TurnStatus.CANCELLED)
        return None

    def _on_llm_delta(self, turn: Turn, kind: str, text: str) -> None:
        if not text:
            return
        event_type = "thinking.delta" if kind == "thinking" else "assistant.delta"
        self.publisher(turn.turn_id, event_type, {"text": text})

    def _loop(self, turn: Turn) -> Turn:
        if turn.status == TurnStatus.FAILED.value:
            return turn
        self.history.save_turn(turn)
        for _ in range(self.config.max_steps):
            before = self.hooks.run_before(BEFORE_LLM_CALL, self._hook_ctx(BEFORE_LLM_CALL, turn))
            if before.is_rejected:
                status = TurnStatus.CANCELLED if before.reason == "cancelled" else TurnStatus.FAILED
                return self._finish(turn, status, before.reason)
            if self.context.should_compact(turn.session_id):
                self._compact(turn)
            response = self._call_llm(turn)
            self.hooks.run_after(AFTER_LLM_CALL, self._hook_ctx(AFTER_LLM_CALL, turn, extra={"usage": response.usage}))
            if response.tool_calls:
                pending = self._handle_tools(turn, response.tool_calls, thinking=response.thinking)
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

    def _call_llm(self, turn: Turn):
        messages = self.context.messages(
            turn.session_id,
            self.extra_system,
            self.user_settings.soul_md,
        )
        return self.llm.complete(
            messages,
            self.schemas(),
            on_delta=lambda kind, text, current=turn: self._on_llm_delta(current, kind, text),
        )

    def _handle_tools(self, turn: Turn, calls: list[ToolCall], *, replay: bool = False, thinking: str = "") -> Turn | None:
        wait_ids: list[str] = []
        for call in calls:
            if not replay:
                self.history.emit(
                    "tool_call",
                    session_id=turn.session_id,
                    turn_id=turn.turn_id,
                    agent_id=turn.agent_id,
                    tool_call_id=call.tool_call_id,
                    name=call.name,
                    arguments=call.arguments,
                    thinking=thinking,
                )
            tool = self.tools.get(call.name)
            result = self._before_tool(turn, call, tool)
            if result.is_rejected and result.reason == "cancelled":
                return self._finish(turn, TurnStatus.CANCELLED)
            if result.is_rejected:
                observation = result.observation or Observation(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.name,
                    outcome="fail",
                    summary=result.reason or "rejected",
                    error=result.reason or "rejected",
                )
                self._write_observation(turn, observation)
                continue
            if result.is_pending:
                return self._enter_pending(turn, call, result)
            if result.is_reuse and result.observation is not None:
                self._write_observation(turn, result.observation)
                continue
            if call.name == "spawn":
                wait_ids.append(call.tool_call_id)
                observation = self._spawn(turn, call)
                self._after_tool(turn, call, tool, observation)
                continue
            if call.name == "remember":
                observation = self._remember(turn, call)
                self._after_tool(turn, call, tool, observation)
                continue
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
            self.publisher(
                turn.turn_id,
                "tool.started",
                {"tool_call_id": call.tool_call_id, "name": call.name, "arguments": call.arguments},
            )
            observation = self.executor.run(self, tool, call)
            self._after_tool(turn, call, tool, observation)
        if wait_ids:
            open_ids = [item for item in wait_ids if self._child_still_open(item)]
            if open_ids:
                turn.status = TurnStatus.PENDING.value
                turn.waiting_for = WaitingFor.CHILDREN.value
                turn.wait_ids = open_ids
                self.history.save_turn(turn)
                return turn
        return None

    def _before_tool(self, turn: Turn, call: ToolCall, tool: Tool | None) -> HookResult:
        ctx = self._hook_ctx(BEFORE_TOOL, turn, call=call, tool=tool)
        result = self.hooks.run_before(BEFORE_TOOL, ctx)
        if result.action != "continue":
            return result
        if tool and tool.before_hooks:
            return self.hooks.run_before_list(tool.before_hooks, ctx)
        return result

    def _after_tool(self, turn: Turn, call: ToolCall, tool: Tool | None, observation: Observation) -> None:
        ctx = self._hook_ctx(AFTER_TOOL_CALL, turn, call=call, tool=tool, observation=observation)
        self.hooks.run_after(AFTER_TOOL_CALL, ctx)
        if tool and tool.after_hooks:
            self.hooks.run_after_list(tool.after_hooks, ctx)
        self._write_observation(turn, observation)

    def _enter_pending(self, turn: Turn, call: ToolCall, result: HookResult) -> Turn:
        turn.status = TurnStatus.PENDING.value
        turn.waiting_for = result.waiting_for or WaitingFor.HUMAN.value
        turn.wait_ids = [call.tool_call_id]
        turn.resume_token = new_resume_token()
        self.history.save_turn(turn)
        self.history.emit(
            "permission",
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            agent_id=turn.agent_id,
            tool_call_id=call.tool_call_id,
            tool_name=call.name,
            reason=result.reason,
            human_params_schema=result.human_params_schema,
        )
        self.publisher(
            turn.turn_id,
            result.event_type or "permission.pending",
            {
                "question": result.question,
                "tool_call_id": call.tool_call_id,
                "tool_name": call.name,
                "reason": result.reason,
            },
        )
        return turn

    def _pending_tool_call(self, turn: Turn) -> ToolCall:
        call_id = turn.wait_ids[0] if turn.wait_ids else ""
        for event in reversed(self.history.events(turn.session_id)):
            if event.type == "tool_call" and event.payload.get("tool_call_id") == call_id:
                return ToolCall(
                    tool_call_id=call_id,
                    name=str(event.payload.get("name") or ""),
                    arguments=dict(event.payload.get("arguments") or {}),
                )
        raise RuntimeError("Pending tool call not found in history")

    def _remember(self, turn: Turn, call: ToolCall) -> Observation:
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
        return bind(
            Observation(
                tool_call_id=call.tool_call_id,
                tool_name="remember",
                summary=f"Stored {record.slot}",
                refs=[record.slot],
            ),
            call,
        )

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
            allowed = {"read_file", "search_web", "ask_user", "read_artifact_range"}
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
            hooks=self.hooks,
            executor=self.executor,
        )
        child_turn = child.run(goal)
        self.sub_agent_ids.append(child.session.agent_id)
        self.history.emit(
            "agent_created",
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            agent_id=turn.agent_id,
            parent_agent_id=self.session.agent_id,
            child_agent_id=child.session.agent_id,
            child_session_id=child.session.session_id,
            child_turn_id=child_turn.turn_id,
        )
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

    def _cancel_agent_tree(self) -> None:
        for event in self.history.events(self.session.session_id):
            if event.type != "agent_created":
                continue
            child_turn_id = event.payload.get("child_turn_id")
            if child_turn_id:
                request_cancel(self.store, str(child_turn_id))

    def _write_observation(self, turn: Turn, observation: Observation) -> None:
        self.history.emit(
            "observation",
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            agent_id=turn.agent_id,
            **observation.to_dict(),
        )
        self.publisher(turn.turn_id, "tool.completed", observation.to_dict())

    def _compact(self, turn: Turn) -> None:
        self.hooks.run_before(PRE_COMPACT, self._hook_ctx(PRE_COMPACT, turn))
        compact_llm = LLM(self.config, small=True)
        events = self.history.events(turn.session_id)
        messages = [
            {
                "role": "system",
                "content": "Summarize the conversation for resume. Keep the user goal, constraints, decisions, facts, and next step. Do not invent.",
            },
            {
                "role": "user",
                "content": json.dumps([event.to_dict() for event in events[-40:]], ensure_ascii=False),
            },
        ]
        try:
            summary = compact_llm.complete(messages, tools=None).text.strip()
        except Exception:  # noqa: BLE001
            self.context.note_compact_failed(turn.session_id)
            return
        if not summary:
            self.context.note_compact_failed(turn.session_id)
            return
        self.context.compact(
            turn.session_id,
            turn.turn_id,
            turn.agent_id,
            summary,
            covers_events=self.context.covered_event_ids(turn.session_id),
        )

    def _finish(self, turn: Turn, status: TurnStatus, error: str | None = None) -> Turn:
        turn.status = status.value
        if error:
            turn.final_text = turn.final_text or error
        self.history.save_turn(turn)
        self.publisher(turn.turn_id, f"turn.{turn.status}", {"error": error})
        if status.value in {TurnStatus.COMPLETED.value, TurnStatus.FAILED.value, TurnStatus.CANCELLED.value}:
            self.hooks.run_after(AFTER_TURN, self._hook_ctx(AFTER_TURN, turn))
            self.hooks.run_after(TURN_CLEANUP, self._hook_ctx(TURN_CLEANUP, turn))
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

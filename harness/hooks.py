"""Lifecycle hooks.

Aligned with 05 Hook: Runtime fires deterministic functions at fixed moments.
Before-hooks may continue, reject, or pending. After-hooks only persist or
record — they never reject or pending.

Critical Before-hooks (balance, cancel, approval, params) are sync and
fail-closed on timeout. Persist-memory / metrics After-hooks run in the
background and must not block the loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable

from harness.billing import debit_turn, ensure_balance
from harness.llm import LLM
from harness.store import is_cancelled, resources_key
from harness.types import Observation, ToolCall, Turn, WaitingFor, new_id

if TYPE_CHECKING:
    from harness.runtime import Runtime
    from harness.tools import Tool

log = logging.getLogger(__name__)

CONTINUE = "continue"
REJECT = "reject"
PENDING = "pending"
REUSE = "reuse"

# Fixed lifecycle moments from 05 Hook. Names are constants; Runtime wires them.
SESSION_START = "session_start"
USER_PROMPT_SUBMIT = "user_prompt_submit"
BEFORE_LLM_CALL = "before_llm_call"
AFTER_LLM_CALL = "after_llm_call"
BEFORE_TOOL = "before_tool"
AFTER_TOOL_CALL = "after_tool_call"
PRE_COMPACT = "pre_compact"
AFTER_TURN = "after_turn"
TURN_CLEANUP = "turn_cleanup"
SESSION_END = "session_end"

DEFAULT_HOOK_TIMEOUT = 30

_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="hook")


@dataclass
class HookResult:
    action: str = CONTINUE
    reason: str = ""
    observation: Observation | None = None
    waiting_for: str | None = None
    human_params_schema: dict[str, Any] | None = None
    question: str = ""
    event_type: str = ""

    @classmethod
    def ok(cls) -> HookResult:
        return cls(CONTINUE)

    @classmethod
    def reject(cls, reason: str, observation: Observation | None = None) -> HookResult:
        return cls(REJECT, reason=reason, observation=observation)

    @classmethod
    def pending(
        cls,
        *,
        waiting_for: str = WaitingFor.HUMAN.value,
        reason: str = "pending_human_params",
        schema: dict[str, Any] | None = None,
        question: str = "",
        event_type: str = "permission.pending",
    ) -> HookResult:
        return cls(
            PENDING,
            reason=reason,
            waiting_for=waiting_for,
            human_params_schema=schema,
            question=question,
            event_type=event_type,
        )

    @classmethod
    def reuse(cls, observation: Observation) -> HookResult:
        return cls(REUSE, reason="idempotent", observation=observation)

    @property
    def is_rejected(self) -> bool:
        return self.action == REJECT

    @property
    def is_pending(self) -> bool:
        return self.action == PENDING

    @property
    def is_reuse(self) -> bool:
        return self.action == REUSE


@dataclass
class HookContext:
    event: str
    runtime: Runtime
    turn: Turn | None = None
    call: ToolCall | None = None
    tool: Tool | None = None
    observation: Observation | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hook:
    name: str
    events: tuple[str, ...]
    run: Callable[[HookContext], HookResult | None]
    sync: bool = True


class HookBus:
    """Ordered hook runner. Before-hooks may stop the loop; After-hooks never do."""

    def __init__(self, timeout_seconds: int = DEFAULT_HOOK_TIMEOUT) -> None:
        self._hooks: dict[str, list[Hook]] = {}
        self.timeout_seconds = timeout_seconds

    def register(self, hook: Hook) -> None:
        for event in hook.events:
            self._hooks.setdefault(event, []).append(hook)

    def run_before(self, event: str, ctx: HookContext) -> HookResult:
        ctx.event = event
        result = HookResult.ok()
        for hook in self._hooks.get(event, []):
            if not hook.sync:
                continue
            returned = self._call_sync(hook, ctx, fail_closed=True)
            if returned is None:
                continue
            if returned.action in {REJECT, PENDING, REUSE}:
                return returned
            result = returned
        return result

    def run_after(self, event: str, ctx: HookContext) -> None:
        """After-hooks persist/record only. Reject/pending from them is ignored."""
        ctx.event = event
        self.run_after_list(self._hooks.get(event, []), ctx)

    def dispatch_async(self, event: str, ctx: HookContext) -> None:
        """Start only non-blocking hooks registered for this lifecycle event."""
        ctx.event = event
        for hook in self._hooks.get(event, []):
            if not hook.sync:
                self._spawn_async(hook, ctx)

    def run_before_list(self, hooks: list[Hook], ctx: HookContext) -> HookResult:
        result = HookResult.ok()
        for hook in hooks:
            if not hook.sync:
                continue
            returned = self._call_sync(hook, ctx, fail_closed=True)
            if returned is None:
                continue
            if returned.action in {REJECT, PENDING, REUSE}:
                return returned
            result = returned
        return result

    def run_after_list(self, hooks: list[Hook], ctx: HookContext) -> None:
        for hook in hooks:
            if hook.sync:
                self._call_sync(hook, ctx, fail_closed=False)
            else:
                self._spawn_async(hook, ctx)

    def _call_sync(self, hook: Hook, ctx: HookContext, *, fail_closed: bool) -> HookResult | None:
        future = _pool.submit(hook.run, ctx)
        try:
            return future.result(timeout=self.timeout_seconds)
        except FuturesTimeout:
            future.cancel()
            log.warning("hook %s timed out after %ss", hook.name, self.timeout_seconds)
            if fail_closed:
                return HookResult.reject("hook_timeout")
            return None
        except Exception:
            log.exception("hook %s failed", hook.name)
            if fail_closed:
                return HookResult.reject("hook_error")
            return None

    def _spawn_async(self, hook: Hook, ctx: HookContext) -> None:
        def _run() -> None:
            try:
                hook.run(ctx)
            except Exception:
                log.exception("async hook %s failed", hook.name)

        Thread(target=_run, name=f"hook-{hook.name}", daemon=True).start()


def session_hooks() -> list[Hook]:
    return [
        Hook("ensure_balance", (SESSION_START,), _ensure_balance_hook),
        Hook("balance", (USER_PROMPT_SUBMIT,), _balance_hook),
        Hook("session_title", (USER_PROMPT_SUBMIT,), _session_title_hook, sync=False),
        Hook("cancel", (BEFORE_LLM_CALL, BEFORE_TOOL), _cancel_hook),
        Hook("schema", (BEFORE_TOOL,), _schema_hook),
        Hook("metrics_llm", (AFTER_LLM_CALL,), _metrics_llm, sync=False),
        Hook("metrics_tool", (AFTER_TOOL_CALL,), _metrics_tool, sync=False),
        Hook("persist_memory", (AFTER_TURN,), _persist_memory, sync=False),
        Hook("metrics_turn", (AFTER_TURN,), _metrics_turn, sync=False),
    ]


def default_hooks(timeout_seconds: int = DEFAULT_HOOK_TIMEOUT) -> HookBus:
    bus = HookBus(timeout_seconds=timeout_seconds)
    for hook in session_hooks():
        bus.register(hook)
    return bus


def interrupt_hook() -> Hook:
    return Hook("interrupt", (BEFORE_TOOL,), _interrupt_hook)


def approval_hook() -> Hook:
    return Hook("approval", (BEFORE_TOOL,), _approval_hook)


def idempotency_hooks() -> tuple[Hook, Hook]:
    return (
        Hook("idempotency", (BEFORE_TOOL,), _idempotency_before),
        Hook("idempotency_after", (AFTER_TOOL_CALL,), _idempotency_after),
    )


def _ensure_balance_hook(ctx: HookContext) -> HookResult | None:
    db = ctx.runtime.config.db()
    if db is None:
        return HookResult.reject("supabase_required")
    ensure_balance(db, ctx.runtime.session.user_id)
    return None


def _balance_hook(ctx: HookContext) -> HookResult | None:
    db = ctx.runtime.config.db()
    if db is None:
        return HookResult.reject("supabase_required")
    ok, remaining = debit_turn(db, ctx.runtime.session.user_id)
    if not ok:
        return HookResult.reject(f"insufficient_balance:{remaining}")
    return None


def _session_title_hook(ctx: HookContext) -> None:
    if ctx.turn is None or ctx.runtime.session.title:
        return
    user_text = str(ctx.extra.get("user_text") or "").strip()
    if not user_text:
        return
    user_messages = [
        event for event in ctx.runtime.history.events(ctx.runtime.session.session_id)
        if event.type == "user_message"
    ]
    if len(user_messages) != 1:
        return
    if not ctx.runtime.config.small_api_key:
        return
    title_llm = LLM(ctx.runtime.config, small=True, reasoning_effort="low")
    response = title_llm.complete(
        [
            {
                "role": "system",
                "content": (
                    "Create a very short title for the user's task. Follow the user's language. "
                    "Chinese: at most 12 characters. English: at most 8 words. "
                    "Return one plain-text line only. No Markdown, quotes, labels, or trailing punctuation. "
                    "Describe the task; do not answer it."
                ),
            },
            {"role": "user", "content": user_text},
        ],
        tools=None,
        temperature=0.1,
    )
    title = sanitize_session_title(response.text, user_text)
    if not title:
        return
    current = ctx.runtime.history.load_session(ctx.runtime.session.session_id)
    if current is None or current.title:
        return
    current.title = title
    ctx.runtime.history.save_session(current)
    ctx.runtime.session.title = title
    ctx.runtime.publisher(
        ctx.turn.turn_id,
        "session.title_updated",
        {"session_id": current.session_id, "title": title, "model": title_llm.model},
    )


def sanitize_session_title(raw: str, user_text: str) -> str:
    title = str(raw or "").splitlines()[0].strip()
    title = re.sub(r"^\s*(?:title|标题)\s*[:：-]\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^#{1,6}\s*", "", title)
    title = title.replace("`", "").replace("**", "").replace("__", "")
    title = title.strip(" \t\"'“”‘’《》【】[]()")
    title = re.sub(r"[。！？!?；;：:，,、.]+$", "", title).strip()
    if not title:
        return ""
    if re.search(r"[\u3400-\u9fff]", user_text):
        title = re.sub(r"\s+", "", title)
        return title[:12].rstrip("。！？!?；;：:，,、.")
    words = title.split()
    return " ".join(words[:8]).rstrip(".?!,;:")


def _cancel_hook(ctx: HookContext) -> HookResult | None:
    if ctx.turn is None:
        return None
    runtime = ctx.runtime
    if runtime.cancelled or is_cancelled(runtime.store, ctx.turn.turn_id):
        return HookResult.reject("cancelled")
    return None


def _schema_hook(ctx: HookContext) -> HookResult | None:
    tool = ctx.tool
    call = ctx.call
    if tool is None or call is None:
        return None
    required = list((tool.parameters or {}).get("required") or [])
    missing = [key for key in required if key not in (call.arguments or {})]
    if missing:
        return HookResult.reject(
            "invalid_params",
            Observation(
                tool_call_id=call.tool_call_id,
                tool_name=call.name,
                outcome="fail",
                summary=f"Missing required arguments: {', '.join(missing)}",
                error="invalid_params",
            ),
        )
    return None


def _interrupt_hook(ctx: HookContext) -> HookResult | None:
    call = ctx.call
    if call is None:
        return None
    answer = (call.human_params or {}).get("answer")
    if answer is None or str(answer).strip() == "":
        question = str((call.arguments or {}).get("question") or "")
        return HookResult.pending(
            reason="pending_human_params",
            schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            question=question,
            event_type="ask_user",
        )
    return None


def _approval_hook(ctx: HookContext) -> HookResult | None:
    call = ctx.call
    if call is None:
        return None
    params = call.human_params or {}
    if "approve" not in params:
        return HookResult.pending(
            reason="pending_approval",
            schema={
                "type": "object",
                "properties": {"approve": {"type": "boolean"}},
                "required": ["approve"],
            },
            question=f"Approve {call.name}?",
            event_type="permission.pending",
        )
    if not _truthy(params.get("approve")):
        return HookResult.reject(
            "user_denied",
            Observation(
                tool_call_id=call.tool_call_id,
                tool_name=call.name,
                outcome="fail",
                summary=f"{call.name} denied by user",
                error="user_denied",
            ),
        )
    return None


def _write_key(turn: Turn, call: ToolCall) -> str:
    payload = json.dumps(
        {"tool": call.name, "arguments": call.arguments, "human": call.human_params},
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return f"turn:{turn.turn_id}:tool:idem:{digest}"


def _idempotency_before(ctx: HookContext) -> HookResult | None:
    call = ctx.call
    if call is None or ctx.turn is None:
        return None
    runtime = ctx.runtime
    key = _write_key(ctx.turn, call)
    claimed = runtime.store.set(key, "running", nx=True, ex=86400)
    runtime.store.sadd(resources_key(ctx.turn.turn_id), key)
    if claimed:
        return None
    raw = runtime.store.get(key)
    if raw and raw not in {"running"}:
        try:
            return HookResult.reuse(Observation.from_dict(json.loads(raw)))
        except json.JSONDecodeError:
            return None
    return HookResult.reuse(
        Observation(
            tool_call_id=call.tool_call_id,
            tool_name=call.name,
            summary="duplicate write skipped",
            preview="idempotent",
        )
    )


def _idempotency_after(ctx: HookContext) -> HookResult | None:
    call = ctx.call
    observation = ctx.observation
    if call is None or observation is None or ctx.turn is None:
        return None
    ctx.runtime.store.set(
        _write_key(ctx.turn, call),
        json.dumps(observation.to_dict(), ensure_ascii=False),
        ex=86400,
    )
    return None


def _metrics_llm(ctx: HookContext) -> None:
    if ctx.turn is None:
        return
    usage = ctx.extra.get("usage") or {}
    payload = {
        "model_id": ctx.runtime.llm.model,
        "reasoning_effort": ctx.runtime.user_settings.reasoning_effort,
        "input_tokens": usage.get("input_tokens") or 0,
        "output_tokens": usage.get("output_tokens") or 0,
    }
    ctx.runtime.history.emit(
        "metrics_llm",
        session_id=ctx.turn.session_id,
        turn_id=ctx.turn.turn_id,
        agent_id=ctx.turn.agent_id,
        **payload,
    )
    ctx.runtime.publisher(ctx.turn.turn_id, "metrics.llm", payload)


def _metrics_tool(ctx: HookContext) -> None:
    if ctx.turn is None or ctx.call is None or ctx.observation is None:
        return
    payload = {
        "tool_call_id": ctx.call.tool_call_id,
        "tool_name": ctx.call.name,
        "status": ctx.observation.outcome,
    }
    ctx.runtime.history.emit(
        "metrics_tool",
        session_id=ctx.turn.session_id,
        turn_id=ctx.turn.turn_id,
        agent_id=ctx.turn.agent_id,
        **payload,
    )
    ctx.runtime.publisher(ctx.turn.turn_id, "metrics.tool", payload)


def _persist_memory(ctx: HookContext) -> None:
    records = ctx.runtime.memory.active()
    if records:
        ctx.runtime.memory.save(records)


def _metrics_turn(ctx: HookContext) -> None:
    if ctx.turn is None:
        return
    ctx.runtime.publisher(ctx.turn.turn_id, "metrics.turn", {"status": ctx.turn.status})


def parse_resume_answer(tool_name: str, answer: str) -> dict[str, Any]:
    if tool_name in {"write_file", "bash"}:
        return {"approve": _truthy(answer)}
    return {"answer": answer}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "approve", "ok"}


def new_resume_token() -> str:
    return new_id("tok")

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from harness.tools import bind
from harness.types import Observation, ToolCall

if TYPE_CHECKING:
    from harness.runtime import Runtime
    from harness.tools import Tool


class ToolExecutor:
    """Runs one Tool Call. Retry lives here; lifecycle hooks do not."""

    def run(self, runtime: Runtime, tool: Tool, call: ToolCall) -> Observation:
        attempts = 0
        last_error: Exception | None = None
        retries = max(0, runtime.config.max_tool_retries)
        while attempts <= retries:
            attempts += 1
            try:
                kwargs = dict(call.arguments)
                if call.human_params is not None:
                    kwargs["_human"] = call.human_params
                return bind(tool.run(**kwargs), call)
            except Exception as exc:  # noqa: BLE001 - tool errors become observations
                last_error = exc
                if attempts > retries or not _retryable(exc):
                    return Observation(
                        tool_call_id=call.tool_call_id,
                        tool_name=call.name,
                        outcome="fail",
                        summary=str(exc),
                        error=type(exc).__name__,
                    )
                time.sleep(min(2 ** (attempts - 1), 4))
        assert last_error is not None
        return Observation(
            tool_call_id=call.tool_call_id,
            tool_name=call.name,
            outcome="fail",
            summary=str(last_error),
            error=type(last_error).__name__,
        )


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return status in {408, 429, 500, 502, 503, 504}

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class TurnStatus(str, Enum):
    RUNNING = "running"
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class WaitingFor(str, Enum):
    HUMAN = "human"
    CHILDREN = "children"


class Outcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"


def _from_dict(cls, data: dict[str, Any]):
    names = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in names})


@dataclass
class Event:
    type: str
    event_id: str = field(default_factory=lambda: new_id("evt"))
    session_id: str = ""
    turn_id: str = ""
    agent_id: str = ""
    created_at: str = field(default_factory=now_iso)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        return _from_dict(cls, data)


@dataclass
class Observation:
    tool_call_id: str
    tool_name: str
    outcome: str = Outcome.PASS.value
    summary: str = ""
    refs: list[str] = field(default_factory=list)
    child_agent_id: str | None = None
    artifact_id: str | None = None
    preview: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        return _from_dict(cls, data)


@dataclass
class ToolCall:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    human_params: dict[str, Any] | None = None


@dataclass
class Turn:
    turn_id: str
    session_id: str
    agent_id: str
    status: str = TurnStatus.RUNNING.value
    waiting_for: str | None = None
    wait_ids: list[str] = field(default_factory=list)
    resume_token: str | None = None
    user_text: str = ""
    final_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Turn:
        return _from_dict(cls, data)


@dataclass
class Session:
    session_id: str
    agent_id: str
    user_id: str
    workspace_id: str
    parent_session_id: str | None = None
    parent_agent_id: str | None = None
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        return _from_dict(cls, data)


@dataclass
class SemanticRecord:
    id: str
    slot: str
    text: str
    layer: str
    status: str
    source_turn_id: str
    source_session_id: str
    valid_from: str = field(default_factory=now_iso)
    retired_at: str | None = None
    updated_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticRecord:
        return _from_dict(cls, data)


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

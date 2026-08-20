from __future__ import annotations

from typing import Any, Iterable

from harness.config import Config
from harness.types import Event, Session, Turn, now_iso


class History:
    def __init__(self, config: Config):
        self.config = config
        self.db = config.db()
        if self.db is None:
            raise RuntimeError("Supabase is required: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")

    def append(self, event: Event) -> Event:
        self.db.insert(
            "events",
            {
                "event_id": event.event_id,
                "type": event.type,
                "session_id": event.session_id,
                "turn_id": event.turn_id,
                "agent_id": event.agent_id,
                "created_at": event.created_at,
                "payload": event.payload,
            },
        )
        return event

    def events(self, session_id: str) -> list[Event]:
        rows = self.db.select(
            "events",
            {"select": "*", "session_id": f"eq.{session_id}", "order": "created_at.asc"},
        )
        return [Event.from_dict(row) for row in rows]

    def save_session(self, session: Session) -> None:
        session.updated_at = now_iso()
        self.db.upsert("sessions", session.to_dict(), "session_id")

    def load_session(self, session_id: str) -> Session | None:
        rows = self.db.select("sessions", {"select": "*", "session_id": f"eq.{session_id}", "limit": "1"})
        return Session.from_dict(rows[0]) if rows else None

    def list_sessions(self, user_id: str, workspace_id: str | None = None) -> list[Session]:
        params: dict[str, str] = {
            "select": "*",
            "user_id": f"eq.{user_id}",
            "order": "updated_at.desc",
            "limit": "200",
        }
        if workspace_id:
            params["workspace_id"] = f"eq.{workspace_id}"
        return [Session.from_dict(row) for row in self.db.select("sessions", params)]

    def list_workspaces(self, user_id: str) -> list[str]:
        rows = self.db.select(
            "sessions",
            {"select": "workspace_id", "user_id": f"eq.{user_id}", "limit": "500"},
        )
        names = []
        seen: set[str] = set()
        for row in rows:
            name = str(row.get("workspace_id") or "default")
            if name not in seen:
                seen.add(name)
                names.append(name)
        return names or ["default"]

    def save_turn(self, turn: Turn) -> None:
        turn.updated_at = now_iso()
        self.db.upsert("turns", turn.to_dict(), "turn_id")

    def load_turn(self, session_id: str) -> Turn | None:
        rows = self.db.select(
            "turns",
            {"select": "*", "session_id": f"eq.{session_id}", "order": "updated_at.desc", "limit": "1"},
        )
        return Turn.from_dict(rows[0]) if rows else None

    def emit(
        self,
        type: str,
        session_id: str,
        turn_id: str = "",
        agent_id: str = "",
        **payload: Any,
    ) -> Event:
        return self.append(
            Event(type=type, session_id=session_id, turn_id=turn_id, agent_id=agent_id, payload=payload)
        )

    def after(self, session_id: str, event_types: Iterable[str]) -> list[Event]:
        wanted = set(event_types)
        return [event for event in self.events(session_id) if event.type in wanted]

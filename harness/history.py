from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from harness.config import Config
from harness.types import Event, Session, Turn


class History:
    def __init__(self, config: Config):
        self.config = config
        config.ensure_dirs()
        self.db = config.db()

    def session_dir(self, session_id: str) -> Path:
        path = self.config.data_dir / "sessions" / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def events_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "events.jsonl"

    def append(self, event: Event) -> Event:
        if self.db:
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
        path = self.events_path(event.session_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return event

    def events(self, session_id: str) -> list[Event]:
        if self.db:
            rows = self.db.select(
                "events",
                {
                    "select": "*",
                    "session_id": f"eq.{session_id}",
                    "order": "created_at.asc",
                },
            )
            return [Event.from_dict(row) for row in rows]
        path = self.events_path(session_id)
        if not path.exists():
            return []
        items: list[Event] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(Event.from_dict(json.loads(line)))
        return items

    def save_session(self, session: Session) -> None:
        if self.db:
            self.db.upsert("sessions", session.to_dict(), "session_id")
            return
        path = self.session_dir(session.session_id) / "session.json"
        path.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_session(self, session_id: str) -> Session | None:
        if self.db:
            rows = self.db.select("sessions", {"select": "*", "session_id": f"eq.{session_id}", "limit": "1"})
            return Session.from_dict(rows[0]) if rows else None
        path = self.session_dir(session_id) / "session.json"
        if not path.exists():
            return None
        return Session.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_turn(self, turn: Turn) -> None:
        row = turn.to_dict()
        if self.db:
            self.db.upsert("turns", row, "turn_id")
            return
        path = self.session_dir(turn.session_id) / "turn.json"
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_turn(self, session_id: str) -> Turn | None:
        if self.db:
            rows = self.db.select(
                "turns",
                {
                    "select": "*",
                    "session_id": f"eq.{session_id}",
                    "order": "updated_at.desc",
                    "limit": "1",
                },
            )
            return Turn.from_dict(rows[0]) if rows else None
        path = self.session_dir(session_id) / "turn.json"
        if not path.exists():
            return None
        return Turn.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def emit(
        self,
        type: str,
        session_id: str,
        turn_id: str = "",
        agent_id: str = "",
        **payload: Any,
    ) -> Event:
        return self.append(
            Event(
                type=type,
                session_id=session_id,
                turn_id=turn_id,
                agent_id=agent_id,
                payload=payload,
            )
        )

    def after(self, session_id: str, event_types: Iterable[str]) -> list[Event]:
        wanted = set(event_types)
        return [event for event in self.events(session_id) if event.type in wanted]

from __future__ import annotations

import json

from harness.config import Config
from harness.types import SemanticRecord, new_id, now_iso


class Memory:
    def __init__(self, config: Config, user_id: str, workspace_id: str):
        self.config = config
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.db = config.db()
        self.path = config.data_dir / "memory" / f"{user_id}__{workspace_id}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[SemanticRecord]:
        if self.db:
            rows = self.db.select(
                "memories",
                {
                    "select": "*",
                    "user_id": f"eq.{self.user_id}",
                    "workspace_id": f"eq.{self.workspace_id}",
                    "order": "updated_at.asc",
                },
            )
            return [SemanticRecord.from_dict(row) for row in rows]
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return [SemanticRecord.from_dict(item) for item in raw]

    def save(self, records: list[SemanticRecord]) -> None:
        if self.db:
            for item in records:
                row = item.to_dict()
                row["user_id"] = self.user_id
                row["workspace_id"] = self.workspace_id
                self.db.upsert("memories", row, "id")
            return
        self.path.write_text(
            json.dumps([item.to_dict() for item in records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def active(self) -> list[SemanticRecord]:
        return [item for item in self.load() if item.status == "active"]

    def prompt_block(self) -> str:
        rows = [item for item in self.active() if item.layer == "profile"]
        if not rows:
            return "No stored profile facts yet."
        return "\n".join(f"- [{item.slot}] {item.text}" for item in rows)

    def upsert(
        self,
        slot: str,
        text: str,
        layer: str,
        source_turn_id: str,
        source_session_id: str,
    ) -> SemanticRecord:
        records = self.load()
        now = now_iso()
        for item in records:
            if item.slot == slot and item.status == "active":
                item.status = "retired"
                item.retired_at = now
                item.updated_at = now
        record = SemanticRecord(
            id=new_id("mem"),
            slot=slot,
            text=text,
            layer=layer,
            status="active",
            source_turn_id=source_turn_id,
            source_session_id=source_session_id,
        )
        records.append(record)
        self.save(records)
        return record

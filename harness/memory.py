from __future__ import annotations

from typing import Any

from harness.config import Config
from harness.types import SemanticRecord, new_id, now_iso


class Memory:
    def __init__(self, config: Config, user_id: str, workspace_id: str):
        self.config = config
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.db = config.db()
        if self.db is None:
            raise RuntimeError("Supabase is required: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")

    def load(self) -> list[SemanticRecord]:
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

    def save(self, records: list[SemanticRecord]) -> None:
        for item in records:
            row = item.to_dict()
            row["user_id"] = self.user_id
            row["workspace_id"] = self.workspace_id
            self.db.upsert("memories", row, "id")

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

    def replace_active(
        self,
        facts: list[dict[str, Any]],
        source_turn_id: str,
        source_session_id: str,
    ) -> None:
        """用 Reflect 后的最终 active 集合替换现有 active 记忆。

        返回集里存在的 slot 保留/更新；不在返回集里的 active 记录被 retire。
        """
        records = self.load()
        now = now_iso()
        final_slots = {
            str(fact.get("slot") or "").strip()
            for fact in facts
            if str(fact.get("slot") or "").strip()
        }
        for item in records:
            if item.status == "active" and item.slot not in final_slots:
                item.status = "retired"
                item.retired_at = now
                item.updated_at = now
        active_by_slot = {item.slot: item for item in records if item.status == "active"}
        for fact in facts:
            slot = str(fact.get("slot") or "").strip()
            text = str(fact.get("text") or "").strip()
            layer = str(fact.get("layer") or "profile").strip() or "profile"
            if layer not in {"profile", "rag"}:
                layer = "profile"
            if not slot or not text:
                continue
            existing = active_by_slot.get(slot)
            if existing is not None:
                existing.text = text
                existing.layer = layer
                existing.updated_at = now
            else:
                records.append(
                    SemanticRecord(
                        id=new_id("mem"),
                        slot=slot,
                        text=text,
                        layer=layer,
                        status="active",
                        source_turn_id=source_turn_id,
                        source_session_id=source_session_id,
                    )
                )
        self.save(records)

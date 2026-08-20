from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from harness.config import Config
from harness.types import now_iso


@dataclass
class UserSettings:
    user_id: str
    default_model: str
    reasoning_effort: str = "medium"
    soul_md: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: Config) -> UserSettings:
        names = {item.name for item in fields(cls)}
        payload = {key: value for key, value in data.items() if key in names}
        payload["default_model"] = str(payload.get("default_model") or config.model)
        payload["reasoning_effort"] = _effort(payload.get("reasoning_effort"))
        payload["soul_md"] = str(payload.get("soul_md") or "")
        payload["settings"] = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
        return cls(**payload)


class UserSettingsStore:
    def __init__(self, config: Config):
        self.config = config
        self.db = config.db()
        if self.db is None:
            raise RuntimeError("Supabase is required: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")

    def get(self, user_id: str) -> UserSettings:
        rows = self.db.select(
            "user_settings",
            {"select": "*", "user_id": f"eq.{user_id}", "limit": "1"},
        )
        if rows:
            return UserSettings.from_dict(rows[0], self.config)
        settings = UserSettings(
            user_id=user_id,
            default_model=self.config.model,
            soul_md=_read_default_soul(self.config.prompts_dir / "SOUL.md"),
        )
        self.save(settings)
        return settings

    def save(self, settings: UserSettings) -> UserSettings:
        settings.reasoning_effort = _effort(settings.reasoning_effort)
        settings.default_model = settings.default_model or self.config.model
        settings.updated_at = now_iso()
        self.db.upsert("user_settings", settings.to_dict(), "user_id")
        return settings

    def update(self, user_id: str, payload: dict[str, Any]) -> UserSettings:
        current = self.get(user_id)
        if "default_model" in payload:
            current.default_model = str(payload.get("default_model") or self.config.model)
        if "reasoning_effort" in payload:
            current.reasoning_effort = _effort(payload.get("reasoning_effort"))
        if "soul_md" in payload:
            current.soul_md = str(payload.get("soul_md") or "")
        extra = payload.get("settings")
        if isinstance(extra, dict):
            current.settings = {**current.settings, **extra}
        return self.save(current)


def _effort(value: Any) -> str:
    effort = str(value or "medium").lower()
    return effort if effort in {"low", "medium", "high"} else "medium"


def _read_default_soul(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()

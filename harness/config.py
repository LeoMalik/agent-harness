from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Config:
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    prompts_dir: Path = ROOT / "prompts"
    skills_dir: Path = ROOT / "skills"
    graphs_dir: Path = ROOT / "graphs"
    model: str = os.getenv("HARNESS_MODEL", "xai/grok-4.5")
    small_model: str = os.getenv("HARNESS_SMALL_MODEL", "deepseek-v4-flash")
    base_url: str = os.getenv("HARNESS_BASE_URL", "https://sub2api-production-3d63.up.railway.app/v1")
    api_key: str = os.getenv("HARNESS_API_KEY", "")
    small_base_url: str = os.getenv("HARNESS_SMALL_BASE_URL", "https://api.deepseek.com/v1")
    small_api_key: str = os.getenv("HARNESS_SMALL_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    redis_url: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    max_steps: int = 16
    max_agent_depth: int = 2
    compact_chars: int = 24_000
    max_inline_chars: int = 8_000
    tool_timeout: int = 30

    def ensure_dirs(self) -> None:
        for path in (self.data_dir / "sessions", self.data_dir / "artifacts", self.data_dir / "memory"):
            path.mkdir(parents=True, exist_ok=True)

from __future__ import annotations

from typing import Any

from harness.types import now_iso

INITIAL_CREDITS = 1_000_000
TURN_COST = 1


def ensure_balance(db: Any, user_id: str, initial: int = INITIAL_CREDITS) -> int:
    rows = db.select("user_balances", {"select": "*", "user_id": f"eq.{user_id}", "limit": "1"})
    if rows:
        return int(rows[0].get("credits") or 0)
    db.upsert(
        "user_balances",
        {"user_id": user_id, "credits": initial, "updated_at": now_iso()},
        "user_id",
    )
    return initial


def debit_turn(db: Any, user_id: str, amount: int = TURN_COST) -> tuple[bool, int]:
    credits = ensure_balance(db, user_id)
    if credits < amount:
        return False, credits
    remaining = credits - amount
    db.upsert(
        "user_balances",
        {"user_id": user_id, "credits": remaining, "updated_at": now_iso()},
        "user_id",
    )
    return True, remaining

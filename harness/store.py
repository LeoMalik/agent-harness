from __future__ import annotations

import json
from typing import Any, Protocol


class Store(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, *, nx: bool = False, ex: int | None = None) -> bool: ...
    def delete(self, key: str) -> None: ...
    def sadd(self, key: str, member: str) -> None: ...
    def smembers(self, key: str) -> set[str]: ...
    def xadd(self, key: str, fields: dict[str, str]) -> str: ...
    def xrange(self, key: str, start: str = "-", end: str = "+", count: int | None = None) -> list[tuple[str, dict[str, str]]]: ...
    def ping(self) -> bool: ...


class MemoryStore:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self._seq = 0

    def get(self, key: str) -> str | None:
        return self.kv.get(key)

    def set(self, key: str, value: str, *, nx: bool = False, ex: int | None = None) -> bool:
        if nx and key in self.kv:
            return False
        self.kv[key] = value
        return True

    def delete(self, key: str) -> None:
        self.kv.pop(key, None)
        self.sets.pop(key, None)
        self.streams.pop(key, None)

    def sadd(self, key: str, member: str) -> None:
        self.sets.setdefault(key, set()).add(member)

    def smembers(self, key: str) -> set[str]:
        return set(self.sets.get(key, set()))

    def xadd(self, key: str, fields: dict[str, str]) -> str:
        self._seq += 1
        event_id = f"{self._seq}-0"
        self.streams.setdefault(key, []).append((event_id, dict(fields)))
        return event_id

    def xrange(self, key: str, start: str = "-", end: str = "+", count: int | None = None) -> list[tuple[str, dict[str, str]]]:
        items = list(self.streams.get(key, []))
        if start != "-":
            items = [item for item in items if item[0] > start]
        if count is not None:
            items = items[:count]
        return items

    def ping(self) -> bool:
        return True


class RedisStore:
    def __init__(self, url: str):
        import redis

        self.client = redis.Redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> str | None:
        value = self.client.get(key)
        return None if value is None else str(value)

    def set(self, key: str, value: str, *, nx: bool = False, ex: int | None = None) -> bool:
        result = self.client.set(key, value, nx=nx, ex=ex)
        return bool(result)

    def delete(self, key: str) -> None:
        self.client.delete(key)

    def sadd(self, key: str, member: str) -> None:
        self.client.sadd(key, member)

    def smembers(self, key: str) -> set[str]:
        return {str(item) for item in self.client.smembers(key)}

    def xadd(self, key: str, fields: dict[str, str]) -> str:
        return str(self.client.xadd(key, fields))

    def xrange(self, key: str, start: str = "-", end: str = "+", count: int | None = None) -> list[tuple[str, dict[str, str]]]:
        rows = self.client.xrange(key, min=start, max=end, count=count)
        return [(str(event_id), {str(k): str(v) for k, v in fields.items()}) for event_id, fields in rows]

    def ping(self) -> bool:
        return bool(self.client.ping())


def connect_store(url: str) -> Store:
    if not url or url == "memory://":
        return MemoryStore()
    try:
        store = RedisStore(url)
        store.ping()
        return store
    except Exception:
        return MemoryStore()


def cancel_key(turn_id: str) -> str:
    return f"turn:{turn_id}:cancelled"


def resources_key(turn_id: str) -> str:
    return f"turn:{turn_id}:resources"


def events_key(turn_id: str) -> str:
    return f"turn:{turn_id}:events"


def request_cancel(store: Store, turn_id: str, ttl: int = 86400) -> None:
    store.set(cancel_key(turn_id), "1", ex=ttl)


def is_cancelled(store: Store, turn_id: str) -> bool:
    return store.get(cancel_key(turn_id)) == "1"


def publish_event(store: Store, turn_id: str, event_type: str, payload: dict[str, Any]) -> str:
    return store.xadd(
        events_key(turn_id),
        {"type": event_type, "payload": json.dumps(payload, ensure_ascii=False)},
    )


def cleanup_turn(store: Store, turn_id: str) -> None:
    for key in store.smembers(resources_key(turn_id)):
        store.delete(key)
    store.delete(resources_key(turn_id))
    store.delete(cancel_key(turn_id))

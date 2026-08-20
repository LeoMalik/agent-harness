from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from harness.billing import ensure_balance
from harness.config import Config
from harness.history import History
from harness.runtime import Runtime
from harness.store import connect_store, events_key
from harness.user_settings import UserSettingsStore


def _json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8") or "{}")


def _turn_payload(runtime: Runtime, turn) -> dict:
    return {
        "session_id": runtime.session.session_id,
        "turn_id": turn.turn_id,
        "status": turn.status,
        "waiting_for": turn.waiting_for,
        "wait_ids": turn.wait_ids,
        "resume_token": turn.resume_token,
        "final_text": turn.final_text,
        "model": runtime.llm.model,
        "reasoning_effort": runtime.user_settings.reasoning_effort,
    }


def _session_payload(session) -> dict:
    return {
        "session_id": session.session_id,
        "workspace_id": session.workspace_id,
        "title": session.title or "Untitled",
        "starred": session.starred,
        "archived": session.archived,
        "trashed": session.trashed,
        "unread": session.unread,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


def _filter_sessions(sessions, name: str):
    if name == "starred":
        return [item for item in sessions if item.starred and not item.trashed]
    if name == "archive":
        return [item for item in sessions if item.archived and not item.trashed]
    if name == "trash":
        return [item for item in sessions if item.trashed]
    if name == "unread":
        return [item for item in sessions if item.unread and not item.trashed]
    if name == "all":
        return list(sessions)
    return [item for item in sessions if not item.archived and not item.trashed]


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        try:
            if path in {"/", "/api", "/api/handshake", "/handshake"}:
                self._handshake()
                return
            if path == "/api/events":
                self._events(query)
                return
            if path == "/api/workspaces":
                self._workspaces(query)
                return
            if path == "/api/sessions":
                self._sessions(query)
                return
            if path == "/api/history":
                self._history(query)
                return
            if path == "/api/stats":
                self._stats(query)
                return
            if path == "/api/settings":
                self._settings(query)
                return
            _json(self, 404, {"error": "not_found"})
        except Exception as exc:  # noqa: BLE001
            _json(self, 500, {"error": type(exc).__name__, "message": str(exc)[:500]})

    def do_POST(self) -> None:
        path = urlparse(self.path).path.rstrip("/")
        payload = _read_json(self)
        cwd = Path("/tmp/harness-workspace")
        cwd.mkdir(parents=True, exist_ok=True)
        try:
            if path == "/api/chat":
                runtime = Runtime.create(
                    Config(),
                    user_id=str(payload.get("user_id") or "local"),
                    workspace_id=str(payload.get("workspace_id") or "default"),
                    cwd=cwd,
                    session_id=payload.get("session_id"),
                )
                turn = runtime.run(str(payload.get("text") or ""))
                _json(self, 200, _turn_payload(runtime, turn))
                return
            if path == "/api/resume":
                session_id = str(payload.get("session_id") or "")
                runtime = Runtime.create(Config(), cwd=cwd, session_id=session_id)
                turn = runtime.resume(session_id, str(payload.get("answer") or ""))
                _json(self, 200, _turn_payload(runtime, turn))
                return
            if path == "/api/cancel":
                self._cancel(payload, cwd)
                return
            if path == "/api/settings":
                user_id = str(payload.get("user_id") or "local")
                settings = UserSettingsStore(Config()).update(user_id, payload)
                _json(self, 200, {"settings": settings.to_dict()})
                return
            if path == "/api/session":
                self._update_session(payload)
                return
            _json(self, 404, {"error": "not_found"})
        except Exception as exc:  # noqa: BLE001
            _json(self, 500, {"error": type(exc).__name__, "message": str(exc)[:500]})

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _handshake(self) -> None:
        config = Config()
        checks = {
            "ok": True,
            "service": "agent-harness",
            "handshake": config.handshake_token,
            "supabase": False,
            "redis": "memory",
        }
        try:
            db = config.db()
            if db:
                db.ping()
                checks["supabase"] = True
        except Exception as exc:  # noqa: BLE001
            checks["ok"] = False
            checks["supabase_error"] = str(exc)
        try:
            store = connect_store(config.redis_url)
            store.ping()
            checks["redis"] = "redis" if config.redis_url and not config.redis_url.startswith("memory") else "memory"
        except Exception as exc:  # noqa: BLE001
            checks["ok"] = False
            checks["redis_error"] = str(exc)
        _json(self, 200 if checks["ok"] else 503, checks)

    def _workspaces(self, query: dict) -> None:
        user_id = (query.get("user_id") or ["local"])[0]
        workspaces = History(Config()).list_workspaces(user_id)
        _json(self, 200, {"workspaces": workspaces})

    def _sessions(self, query: dict) -> None:
        user_id = (query.get("user_id") or ["local"])[0]
        workspace_id = (query.get("workspace_id") or [None])[0]
        filter_name = (query.get("filter") or ["all"])[0]
        sessions = History(Config()).list_sessions(user_id, workspace_id)
        items = [_session_payload(item) for item in _filter_sessions(sessions, filter_name)]
        _json(self, 200, {"sessions": items})

    def _history(self, query: dict) -> None:
        session_id = (query.get("session_id") or [""])[0]
        if not session_id:
            _json(self, 200, {"events": []})
            return
        events = History(Config()).events(session_id)
        allowed = {"user_message", "assistant_message", "tool_call", "observation", "permission"}
        items = [event.to_dict() for event in events if event.type in allowed]
        _json(self, 200, {"events": items})

    def _settings(self, query: dict) -> None:
        user_id = (query.get("user_id") or ["local"])[0]
        config = Config()
        settings = UserSettingsStore(config).get(user_id)
        _json(
            self,
            200,
            {
                "settings": settings.to_dict(),
                "available_models": [config.model, config.small_model],
            },
        )

    def _stats(self, query: dict) -> None:
        session_id = (query.get("session_id") or [""])[0]
        user_id = (query.get("user_id") or ["local"])[0]
        config = Config()
        history = History(config)
        session = history.load_session(session_id) if session_id else None
        if session is not None:
            user_id = session.user_id
        events = history.events(session_id) if session_id else []
        metrics = [event.payload for event in events if event.type == "metrics_llm"]
        input_tokens = sum(int(item.get("input_tokens") or 0) for item in metrics)
        output_tokens = sum(int(item.get("output_tokens") or 0) for item in metrics)
        tool_calls = sum(1 for event in events if event.type == "tool_call")
        turns = config.db().select(
            "turns",
            {"select": "turn_id", "session_id": f"eq.{session_id}", "limit": "500"},
        ) if session_id else []
        settings = UserSettingsStore(config).get(user_id)
        credits = ensure_balance(config.db(), user_id)
        latest_turn = history.load_turn(session_id) if session_id else None
        _json(
            self,
            200,
            {
                "turns": len(turns),
                "tool_calls": tool_calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "credits": credits,
                "model": metrics[-1].get("model_id") if metrics else settings.default_model,
                "reasoning_effort": metrics[-1].get("reasoning_effort") if metrics else settings.reasoning_effort,
                "status": latest_turn.status if latest_turn else "ready",
            },
        )

    def _update_session(self, payload: dict) -> None:
        session_id = str(payload.get("session_id") or "")
        history = History(Config())
        session = history.load_session(session_id)
        if session is None:
            _json(self, 404, {"error": "session_not_found"})
            return
        for key in ("title", "starred", "archived", "trashed", "unread", "workspace_id"):
            if key in payload:
                setattr(session, key, payload[key])
        history.save_session(session)
        _json(self, 200, _session_payload(session))

    def _cancel(self, payload: dict, cwd: Path) -> None:
        session_id = str(payload.get("session_id") or "")
        runtime = Runtime.create(Config(), cwd=cwd, session_id=session_id)
        turn = runtime.history.load_turn(session_id)
        if turn is None:
            _json(self, 404, {"error": "no_turn"})
            return
        runtime.cancel(turn.turn_id)
        latest = runtime.history.load_turn(session_id) or turn
        _json(self, 200, _turn_payload(runtime, latest))

    def _events(self, query: dict) -> None:
        turn_id = (query.get("turn_id") or [""])[0]
        session_id = (query.get("session_id") or [""])[0]
        if session_id and not turn_id:
            turn = History(Config()).load_turn(session_id)
            if turn:
                turn_id = turn.turn_id
        if not turn_id:
            _json(self, 200, {"turn_id": None, "events": []})
            return
        store = connect_store(Config().redis_url)
        events = []
        for event_id, fields in store.xrange(events_key(turn_id)):
            raw = fields.get("payload", "{}")
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                payload = {"raw": raw}
            events.append({"id": event_id, "type": fields.get("type", ""), "payload": payload})
        _json(self, 200, {"turn_id": turn_id, "events": events})

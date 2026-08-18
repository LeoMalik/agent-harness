from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from harness.config import Config
from harness.runtime import Runtime
from harness.store import request_cancel


def _json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
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
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in {"/", "/api", "/api/handshake", "/handshake"}:
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
                from harness.store import connect_store

                store = connect_store(config.redis_url)
                store.ping()
                checks["redis"] = "redis" if config.redis_url and not config.redis_url.startswith("memory") else "memory"
            except Exception as exc:  # noqa: BLE001
                checks["ok"] = False
                checks["redis_error"] = str(exc)
            _json(self, 200 if checks["ok"] else 503, checks)
            return
        _json(self, 404, {"error": "not_found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path.rstrip("/")
        payload = _read_json(self)
        cwd = Path("/tmp/harness-workspace")
        cwd.mkdir(parents=True, exist_ok=True)
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
            session_id = str(payload.get("session_id") or "")
            runtime = Runtime.create(Config(), cwd=cwd, session_id=session_id)
            turn = runtime.history.load_turn(session_id)
            if turn is None:
                _json(self, 404, {"error": "no_turn"})
                return
            request_cancel(runtime.store, turn.turn_id)
            runtime.cancel(turn.turn_id)
            latest = runtime.history.load_turn(session_id) or turn
            _json(self, 200, _turn_payload(runtime, latest))
            return
        _json(self, 404, {"error": "not_found"})

from __future__ import annotations

from typing import Any

import httpx


class Supabase:
    def __init__(self, url: str, service_role_key: str):
        self.url = url.rstrip("/")
        self.headers = {
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        with httpx.Client(timeout=30) as client:
            response = client.request(
                method,
                f"{self.url}/rest/v1/{path}",
                headers={**self.headers, **kwargs.pop("headers", {})},
                **kwargs,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"Supabase {response.status_code}: {response.text[:800]}")
            if not response.content:
                return None
            return response.json()

    def select(self, table: str, params: dict[str, str]) -> list[dict[str, Any]]:
        return self._request("GET", table, params=params) or []

    def insert(self, table: str, row: dict[str, Any]) -> list[dict[str, Any]]:
        return self._request("POST", table, json=row, headers={"Prefer": "return=representation"}) or []

    def upsert(self, table: str, row: dict[str, Any], on_conflict: str) -> list[dict[str, Any]]:
        return (
            self._request(
                "POST",
                table,
                json=row,
                headers={"Prefer": "return=representation,resolution=merge-duplicates"},
                params={"on_conflict": on_conflict},
            )
            or []
        )

    def patch(self, table: str, params: dict[str, str], row: dict[str, Any]) -> list[dict[str, Any]]:
        return self._request("PATCH", table, params=params, json=row, headers={"Prefer": "return=representation"}) or []

    def storage_put(self, bucket: str, path: str, content: str) -> None:
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{self.url}/storage/v1/object/{bucket}/{path}",
                headers={
                    "apikey": self.headers["apikey"],
                    "Authorization": self.headers["Authorization"],
                    "x-upsert": "true",
                },
                content=content.encode("utf-8"),
            )
            if response.status_code >= 400:
                raise RuntimeError(f"Storage upload {response.status_code}: {response.text[:800]}")

    def storage_get(self, bucket: str, path: str) -> str:
        with httpx.Client(timeout=60) as client:
            response = client.get(
                f"{self.url}/storage/v1/object/{bucket}/{path}",
                headers={"apikey": self.headers["apikey"], "Authorization": self.headers["Authorization"]},
            )
            if response.status_code >= 400:
                raise RuntimeError(f"Storage download {response.status_code}: {response.text[:800]}")
            return response.text

    def storage_list(self, bucket: str, prefix: str = "") -> list[str]:
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{self.url}/storage/v1/object/list/{bucket}",
                headers={
                    "apikey": self.headers["apikey"],
                    "Authorization": self.headers["Authorization"],
                    "Content-Type": "application/json",
                },
                json={"prefix": prefix, "limit": 1000},
            )
            if response.status_code >= 400:
                raise RuntimeError(f"Storage list {response.status_code}: {response.text[:800]}")
            return [item.get("name", "") for item in response.json()]

    def ping(self) -> bool:
        self.select("sessions", {"select": "session_id", "limit": "1"})
        return True

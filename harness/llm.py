from __future__ import annotations

import json
from typing import Any

import httpx

from harness.config import Config
from harness.types import ModelResponse, ToolCall, new_id


class LLM:
    def __init__(self, config: Config, small: bool = False):
        self.config = config
        self.small = small

    @property
    def model(self) -> str:
        return self.config.small_model if self.small else self.config.model

    @property
    def base_url(self) -> str:
        if self.small and self.config.small_base_url:
            return self.config.small_base_url.rstrip("/")
        return self.config.base_url.rstrip("/")

    @property
    def api_key(self) -> str:
        if self.small and self.config.small_api_key:
            return self.config.small_api_key
        return self.config.api_key

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> ModelResponse:
        if not self.api_key:
            raise RuntimeError("HARNESS_API_KEY is not set")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        with httpx.Client(timeout=120) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"LLM {response.status_code} model={self.model} url={self.base_url}: {response.text[:800]}"
                )
            data = response.json()
        message = data["choices"][0]["message"]
        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            arguments = raw.get("function", {}).get("arguments") or "{}"
            if isinstance(arguments, str):
                arguments = json.loads(arguments or "{}")
            calls.append(
                ToolCall(
                    tool_call_id=raw.get("id") or new_id("call"),
                    name=raw.get("function", {}).get("name", ""),
                    arguments=arguments,
                )
            )
        usage = data.get("usage") or {}
        return ModelResponse(
            text=message.get("content") or "",
            tool_calls=calls,
            usage={
                "input_tokens": int(usage.get("prompt_tokens") or 0),
                "output_tokens": int(usage.get("completion_tokens") or 0),
            },
        )

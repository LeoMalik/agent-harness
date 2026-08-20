from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from harness.config import Config
from harness.types import ModelResponse, ToolCall, new_id

DeltaFn = Callable[[str, str], None]


class LLM:
    def __init__(
        self,
        config: Config,
        small: bool = False,
        model_override: str | None = None,
        reasoning_effort: str = "medium",
    ):
        self.config = config
        self.small = small
        self.model_override = model_override
        self.reasoning_effort = reasoning_effort

    @property
    def _use_small(self) -> bool:
        wants_small = self.small or self.model_override == self.config.small_model
        return wants_small and bool(self.config.small_api_key)

    @property
    def model(self) -> str:
        return self.model_override or (self.config.small_model if self._use_small else self.config.model)

    @property
    def base_url(self) -> str:
        return self.config.small_base_url.rstrip("/") if self._use_small else self.config.base_url.rstrip("/")

    @property
    def api_key(self) -> str:
        return self.config.small_api_key if self._use_small else self.config.api_key

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        on_delta: DeltaFn | None = None,
    ) -> ModelResponse:
        if not self.api_key:
            raise RuntimeError("HARNESS_API_KEY is not set")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "reasoning_effort": self.reasoning_effort,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        try:
            return self._stream(body, on_delta)
        except Exception:
            body["stream"] = False
            return self._once(body, on_delta)

    def _once(self, body: dict[str, Any], on_delta: DeltaFn | None) -> ModelResponse:
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
        thinking = message.get("reasoning_content") or message.get("reasoning") or ""
        text = message.get("content") or ""
        if on_delta and thinking:
            on_delta("thinking", thinking)
        if on_delta and text:
            on_delta("assistant", text)
        return ModelResponse(
            text=text,
            thinking=thinking,
            tool_calls=_parse_tool_calls(message.get("tool_calls") or []),
            usage=_usage(data.get("usage") or {}),
        )

    def _stream(self, body: dict[str, Any], on_delta: DeltaFn | None) -> ModelResponse:
        text_parts: list[str] = []
        think_parts: list[str] = []
        tool_acc: dict[int, dict[str, str]] = {}
        usage: dict[str, int] = {}
        with httpx.Client(timeout=120) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            ) as response:
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"LLM {response.status_code} model={self.model} url={self.base_url}: {response.read()[:800]!r}"
                    )
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    chunk = json.loads(raw)
                    if chunk.get("usage"):
                        usage = _usage(chunk["usage"])
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        think = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        content = delta.get("content") or ""
                        if think:
                            think_parts.append(think)
                            if on_delta:
                                on_delta("thinking", think)
                        if content:
                            text_parts.append(content)
                            if on_delta:
                                on_delta("assistant", content)
                        for call in delta.get("tool_calls") or []:
                            index = int(call.get("index") or 0)
                            slot = tool_acc.setdefault(index, {"id": "", "name": "", "arguments": ""})
                            if call.get("id"):
                                slot["id"] = call["id"]
                            fn = call.get("function") or {}
                            if fn.get("name"):
                                slot["name"] += fn["name"]
                            if fn.get("arguments"):
                                slot["arguments"] += fn["arguments"]
        calls = []
        for slot in tool_acc.values():
            arguments = slot["arguments"] or "{}"
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = {}
            calls.append(
                ToolCall(
                    tool_call_id=slot["id"] or new_id("call"),
                    name=slot["name"],
                    arguments=parsed,
                )
            )
        return ModelResponse(
            text="".join(text_parts),
            thinking="".join(think_parts),
            tool_calls=calls,
            usage=usage,
        )


def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for raw in raw_calls:
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
    return calls


def _usage(usage: dict[str, Any]) -> dict[str, int]:
    return {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
    }

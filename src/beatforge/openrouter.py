"""Minimal OpenRouter transport with an explicit free-only development guard."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Callable
from urllib import error, request
from urllib.parse import urlsplit

from .providers import OPENROUTER_FREE_ROUTER_ID, is_free_model_id, parse_free_model


@dataclass(frozen=True, slots=True)
class ChatResult:
    content: str
    model: str
    usage: dict[str, Any]
    raw: dict[str, Any]
    tool_calls: tuple[dict[str, Any], ...] = ()


class OpenRouterResponseError(RuntimeError):
    """Provider returned a response that cannot form a model ChatResult."""

    def __init__(self, message: str, *, usage: dict[str, Any] | None = None,
                 finish_reason: str | None = None,
                 diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.usage = dict(usage or {})
        self.finish_reason = finish_reason
        self.diagnostics = dict(diagnostics or {})


def _diagnostic_text(value: Any, *, limit: int = 300) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[redacted]", value.strip())
    text = re.sub(r"(?i)(api[_ -]?key\s*[:=]\s*)[^\s,;]+", r"\1[redacted]", text)
    return text[:limit]


def _provider_diagnostics(value: dict[str, Any]) -> dict[str, Any]:
    """Keep a small provider-error summary without retaining model output or raw JSON."""
    result: dict[str, Any] = {}
    model = value.get("model")
    if isinstance(model, str) and model:
        result["selected_model"] = model[:160]

    error_value = value.get("error")
    if isinstance(error_value, dict):
        summary: dict[str, Any] = {}
        code = error_value.get("code")
        if isinstance(code, (str, int)) and not isinstance(code, bool):
            summary["code"] = str(code)[:80]
        error_type = error_value.get("type")
        if isinstance(error_type, str) and error_type:
            summary["type"] = error_type[:120]
        message = _diagnostic_text(error_value.get("message"))
        if message:
            summary["message"] = message
        metadata = error_value.get("metadata")
        if isinstance(metadata, dict):
            safe_metadata: dict[str, Any] = {}
            for key in ("provider_name", "provider_error_code", "limit_source", "remedy_hint"):
                safe_value = _diagnostic_text(metadata.get(key), limit=240)
                if safe_value:
                    safe_metadata[key] = safe_value
            if isinstance(metadata.get("is_byok"), bool):
                safe_metadata["is_byok"] = metadata["is_byok"]
            raw = metadata.get("raw")
            if isinstance(raw, str):
                try:
                    parsed_raw = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    parsed_raw = None
                if isinstance(parsed_raw, dict):
                    nested_error = parsed_raw.get("error")
                    if isinstance(nested_error, dict):
                        nested_message = _diagnostic_text(nested_error.get("message"))
                        if nested_message:
                            safe_metadata["provider_message"] = nested_message
                        nested_code = nested_error.get("code")
                        if isinstance(nested_code, (str, int)) and not isinstance(nested_code, bool):
                            safe_metadata["provider_code"] = str(nested_code)[:80]
            if safe_metadata:
                summary["metadata"] = safe_metadata
        if summary:
            result["error"] = summary

    choices = value.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        choice_summary: dict[str, Any] = {}
        for key in ("finish_reason", "native_finish_reason"):
            text = _diagnostic_text(choice.get(key), limit=120)
            if text:
                choice_summary[key] = text
        message = choice.get("message")
        if isinstance(message, dict):
            choice_summary["message_fields"] = sorted(str(key)[:80] for key in message)[:32]
            refusal = _diagnostic_text(message.get("refusal"))
            if refusal:
                choice_summary["refusal"] = refusal
        if choice_summary:
            result["choice"] = choice_summary

    routing = value.get("openrouter_metadata")
    if isinstance(routing, dict):
        route: dict[str, Any] = {}
        for key in ("strategy", "attempt"):
            item = routing.get(key)
            if isinstance(item, (str, int, float, bool)):
                route[key] = item
        if isinstance(routing.get("is_byok"), bool):
            route["is_byok"] = routing["is_byok"]
        endpoints = routing.get("endpoints")
        available = endpoints.get("available") if isinstance(endpoints, dict) else None
        selected = next((item for item in available or []
                         if isinstance(item, dict) and item.get("selected") is True), None)
        if selected is not None:
            for source, target in (("model", "selected_model"), ("provider", "selected_provider")):
                item = selected.get(source)
                if isinstance(item, str) and item:
                    route[target] = item[:160]
        if route:
            result["route"] = route
    return result


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, *, base_url: str = "https://openrouter.ai/api/v1",
                 timeout_seconds: float = 120.0, enforce_free: bool = True,
                 require_loopback: bool = False, allow_missing_api_key: bool = False):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.enforce_free = enforce_free
        self.allow_missing_api_key = allow_missing_api_key
        self.require_loopback = require_loopback
        self._progress_callback: Callable[[str], None] | None = None
        self._model_modalities: dict[str, frozenset[str]] = {}
        self._catalog_cache: list[dict[str, Any]] | None = None
        if require_loopback:
            hostname = urlsplit(self.base_url).hostname
            if hostname not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError("local provider endpoint must use loopback")

    def set_progress_callback(self, callback: Callable[[str], None] | None) -> None:
        """Receive cumulative visible response text while a streamed call runs."""
        if callback is not None and not callable(callback):
            raise TypeError("progress callback must be callable or None")
        self._progress_callback = callback

    def get_progress_callback(self) -> Callable[[str], None] | None:
        """Return the callback currently receiving streamed visible text."""
        return self._progress_callback

    def _notify_progress(self, text: str) -> None:
        callback = self._progress_callback
        if callback is not None:
            callback(text)

    def _json_request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_key and not self.allow_missing_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json", "X-Title": "BeatForge"}
        if payload is not None and payload.get("model") == OPENROUTER_FREE_ROUTER_ID:
            headers["X-OpenRouter-Metadata"] = "enabled"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.enforce_free:
            headers["HTTP-Referer"] = "https://localhost/beatforge"
        req = request.Request(self.base_url + path, data=body, headers=headers, method=method)
        if payload is not None and payload.get("stream") is True:
            return self._stream_json_request(req)
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                if self.require_loopback:
                    final_url = getattr(response, "geturl", lambda: self.base_url)()
                    final_host = urlsplit(final_url).hostname
                    if final_host not in {"127.0.0.1", "localhost", "::1"}:
                        raise RuntimeError("local provider redirect escaped loopback")
                value = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail[:500]}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenRouter transport error: {exc.reason}") from exc
        if not isinstance(value, dict):
            raise RuntimeError("OpenRouter returned a non-object response")
        return value

    def _stream_json_request(self, req: request.Request) -> dict[str, Any]:
        """Collect an OpenAI-compatible SSE response and publish visible text deltas."""
        content_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        openrouter_metadata: dict[str, Any] = {}
        model: str | None = None
        finish_reason: str | None = None

        def dispatch(data_lines: list[str]) -> None:
            nonlocal model, usage, finish_reason, openrouter_metadata
            if not data_lines:
                return
            data = "\n".join(data_lines)
            if data == "[DONE]":
                return
            try:
                value = json.loads(data)
            except json.JSONDecodeError as exc:
                raise OpenRouterResponseError("OpenRouter stream contained invalid JSON",
                                              usage=usage) from exc
            if not isinstance(value, dict):
                return
            error_value = value.get("error")
            if isinstance(error_value, dict):
                detail = str(error_value.get("message", "OpenRouter stream failed"))[:400]
                raise OpenRouterResponseError(
                    detail, usage=value.get("usage") or usage,
                    diagnostics=_provider_diagnostics(value),
                )
            if isinstance(value.get("model"), str):
                model = value["model"]
            if isinstance(value.get("usage"), dict):
                usage = value["usage"]
            if isinstance(value.get("openrouter_metadata"), dict):
                openrouter_metadata = value["openrouter_metadata"]
            for choice in value.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                if isinstance(content, str) and content:
                    content_parts.append(content)
                    self._notify_progress("".join(content_parts))
                elif isinstance(content, list):
                    text_parts = [part.get("text", "") for part in content
                                  if isinstance(part, dict) and isinstance(part.get("text"), str)]
                    if text_parts:
                        content_parts.extend(text_parts)
                        self._notify_progress("".join(content_parts))
                for call_delta in delta.get("tool_calls") or []:
                    if not isinstance(call_delta, dict):
                        continue
                    index = call_delta.get("index", 0)
                    if not isinstance(index, int) or isinstance(index, bool):
                        continue
                    call = tool_calls.setdefault(index, {
                        "id": "", "type": "function", "function": {"name": "", "arguments": ""},
                    })
                    if isinstance(call_delta.get("id"), str):
                        call["id"] = call_delta["id"]
                    if isinstance(call_delta.get("type"), str):
                        call["type"] = call_delta["type"]
                    function = call_delta.get("function") or {}
                    if isinstance(function, dict):
                        for key in ("name", "arguments"):
                            value_part = function.get(key)
                            if isinstance(value_part, str):
                                call["function"][key] += value_part

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                if self.require_loopback:
                    final_url = getattr(response, "geturl", lambda: self.base_url)()
                    final_host = urlsplit(final_url).hostname
                    if final_host not in {"127.0.0.1", "localhost", "::1"}:
                        raise RuntimeError("local provider redirect escaped loopback")
                self._notify_progress("")
                data_lines: list[str] = []
                for raw_line in response:
                    decoded = raw_line.decode("utf-8", errors="replace")
                    for line in decoded.splitlines():
                        if not line:
                            dispatch(data_lines)
                            data_lines = []
                        elif line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
                dispatch(data_lines)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail[:500]}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenRouter transport error: {exc.reason}") from exc

        message: dict[str, Any] = {"role": "assistant"}
        if content_parts:
            message["content"] = "".join(content_parts)
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        choice: dict[str, Any] = {"message": message, "finish_reason": finish_reason}
        return {"model": model, "choices": [choice], "usage": usage, "stream": True,
                "openrouter_metadata": openrouter_metadata,
                "provider_diagnostics": _provider_diagnostics({
                    "model": model, "choices": [choice],
                    "openrouter_metadata": openrouter_metadata,
                })}

    def catalog(self) -> list[dict[str, Any]]:
        if self._catalog_cache is not None:
            return list(self._catalog_cache)
        value = self._json_request("GET", "/models")
        rows = value.get("data")
        if not isinstance(rows, list):
            raise RuntimeError("OpenRouter catalog has no data array")
        self._catalog_cache = [row for row in rows if isinstance(row, dict)]
        return list(self._catalog_cache)

    def _free_catalog_entry(self, model_id: str) -> dict[str, Any]:
        row = next((item for item in self.catalog() if item.get("id") == model_id), None)
        if row is None:
            raise LookupError(f"model is absent from OpenRouter catalog: {model_id!r}")
        parse_free_model(row)
        return row

    def supports_input_modality(self, model_id: str, modality: str) -> bool:
        """Check the live catalog and require verified zero pricing for remote dev models."""
        if modality not in {"text", "image", "audio", "video"}:
            raise ValueError(f"unsupported input modality: {modality}")
        if model_id not in self._model_modalities:
            if self.enforce_free:
                row = self._free_catalog_entry(model_id)
            else:
                row = next((item for item in self.catalog() if item.get("id") == model_id), None)
                if row is None:
                    raise LookupError(f"model is absent from OpenRouter catalog: {model_id!r}")
            architecture = row.get("architecture") or {}
            modalities = frozenset(architecture.get("input_modalities") or [])
            self._model_modalities[model_id] = modalities
        return modality in self._model_modalities[model_id]

    def supports_tools(self, model_id: str) -> bool:
        """Use the live model catalog; do not assume tool support from the family name."""
        row = self._free_catalog_entry(model_id) if self.enforce_free else next(
            (item for item in self.catalog() if item.get("id") == model_id), None)
        if row is None:
            raise LookupError(f"model is absent from OpenRouter catalog: {model_id!r}")
        return "tools" in set(row.get("supported_parameters") or [])

    def chat(self, model_id: str, messages: list[dict[str, Any]], *, temperature: float = 0.2,
             reasoning_effort: str | None = None,
             tools: list[dict[str, Any]] | None = None,
             tool_choice: str | dict[str, Any] | None = None,
             parallel_tool_calls: bool | None = None) -> ChatResult:
        if self.enforce_free and not is_free_model_id(model_id):
            raise ValueError("BeatForge development policy permits only OpenRouter :free models or openrouter/free")
        if self.enforce_free:
            self._free_catalog_entry(model_id)
        if not messages:
            raise ValueError("messages cannot be empty")
        payload: dict[str, Any] = {
            "model": model_id, "messages": messages,
            "temperature": temperature,
        }
        if tools:
            if self.enforce_free and not self.supports_tools(model_id):
                raise ValueError(f"model does not advertise tool support: {model_id}")
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
            if parallel_tool_calls is not None:
                payload["parallel_tool_calls"] = parallel_tool_calls
        if reasoning_effort is not None:
            if not isinstance(reasoning_effort, str) or not reasoning_effort.strip():
                raise ValueError("reasoning_effort must be a non-empty string")
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        if self._progress_callback is not None:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        value = self._json_request("POST", "/chat/completions", payload)
        choices = value.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            error_value = value.get("error")
            if isinstance(error_value, dict):
                diagnostics = _provider_diagnostics(value)
                error_summary = diagnostics.get("error", {})
                metadata = error_summary.get("metadata", {})
                provider = metadata.get("provider_name")
                detail = error_summary.get("message", "")
                raise OpenRouterResponseError(
                    "OpenRouter response has no choices "
                    f"(code={error_summary.get('code')}, provider={provider!r}, message={detail!r})",
                    usage=value.get("usage") or {},
                    diagnostics=diagnostics,
                )
            raise OpenRouterResponseError(
                "OpenRouter response has no choices "
                f"(response_fields={sorted(value)})", usage=value.get("usage") or {},
                diagnostics=_provider_diagnostics(value),
            )
        message = choices[0].get("message") or {}
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            tool_calls = []
        valid_calls = tuple(call for call in tool_calls if isinstance(call, dict))
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content
                              if isinstance(part, dict) and isinstance(part.get("text"), str))
        if content is None and valid_calls:
            content = ""
        if not isinstance(content, str) or (not content.strip() and not valid_calls):
            fields = sorted(message) if isinstance(message, dict) else []
            finish = choices[0].get("finish_reason")
            reasoning_returned = bool(message.get("reasoning") or message.get("reasoning_details")) \
                if isinstance(message, dict) else False
            raise OpenRouterResponseError(
                "OpenRouter response has no textual content "
                f"(finish_reason={finish!r}, message_fields={fields}, "
                f"reasoning_returned={reasoning_returned})",
                usage=value.get("usage") or {}, finish_reason=finish,
                diagnostics=value.get("provider_diagnostics")
                if isinstance(value.get("provider_diagnostics"), dict)
                else _provider_diagnostics(value),
            )
        actual_model = str(value.get("model", model_id))
        usage = dict(value.get("usage") or {})
        if model_id == OPENROUTER_FREE_ROUTER_ID:
            routing = value.get("openrouter_metadata")
            endpoints = routing.get("endpoints") if isinstance(routing, dict) else None
            available = endpoints.get("available") if isinstance(endpoints, dict) else None
            selected = next((endpoint for endpoint in available or []
                             if isinstance(endpoint, dict) and endpoint.get("selected") is True), None)
            if selected is not None:
                routed_model = selected.get("model")
                routed_provider = selected.get("provider")
                route = {"requested_model": model_id}
                if isinstance(routed_model, str) and routed_model:
                    route["selected_model"] = routed_model
                    actual_model = routed_model
                if isinstance(routed_provider, str) and routed_provider:
                    route["selected_provider"] = routed_provider
                for key in ("strategy", "attempt"):
                    if key in routing and isinstance(routing[key], (str, int, float, bool)):
                        route[key] = routing[key]
                if "is_byok" in routing and isinstance(routing["is_byok"], bool):
                    route["is_byok"] = routing["is_byok"]
                usage["beatforge_model_route"] = route
        return ChatResult(content, actual_model, usage, value, valid_calls)


class LocalOpenAIClient(OpenRouterClient):
    """Loopback-only OpenAI-compatible transport for llama.cpp and similar servers."""

    def __init__(self, *, base_url: str = "http://127.0.0.1:8080/v1",
                 api_key: str | None = None, timeout_seconds: float = 120.0):
        super().__init__(api_key, base_url=base_url, timeout_seconds=timeout_seconds,
                         enforce_free=False, require_loopback=True,
                         allow_missing_api_key=True)

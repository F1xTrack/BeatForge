import json

import pytest

from beatforge import LocalOpenAIClient, OpenRouterClient


def test_openrouter_rejects_paid_model_before_transport():
    client = OpenRouterClient("key", base_url="https://example.test")
    with pytest.raises(ValueError, match=":free"):
        client.chat("demo/paid", [{"role": "user", "content": "x"}])


def test_openrouter_parses_free_chat(monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(req, **kwargs):
        if req.data is None:
            return Response({"data": [{"id": "demo/model:free",
                                       "pricing": {"prompt": "0", "completion": "0"}}]})
        return Response({"model": "demo/model:free", "choices": [{"message": {"content": "map.note(...)"}}],
                         "usage": {"prompt_tokens": 4}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = OpenRouterClient("key").chat("demo/model:free", [{"role": "user", "content": "x"}])
    assert result.content == "map.note(...)"
    assert result.usage["prompt_tokens"] == 4


def test_openrouter_free_router_checks_zero_price_and_records_routed_model(monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(req, **kwargs):
        if req.data is None:
            return Response({"data": [{"id": "openrouter/free",
                                       "pricing": {"prompt": "0", "completion": "0"},
                                       "architecture": {"input_modalities": ["text", "image"]},
                                       "supported_parameters": ["reasoning", "tools"]}]})
        sent = json.loads(req.data.decode())
        assert sent["model"] == "openrouter/free"
        assert "max_tokens" not in sent
        assert {key.lower(): value for key, value in req.header_items()}[
            "x-openrouter-metadata"] == "enabled"
        return Response({"model": "openrouter/free",
                         "choices": [{"message": {"content": "map.note(...)"}}],
                         "usage": {"cost": 0, "is_byok": False},
                         "openrouter_metadata": {
                             "requested": "openrouter/free", "strategy": "free-models",
                             "is_byok": False,
                             "endpoints": {"available": [{
                                 "model": "dots-studio/dots-3-note-preview:free",
                                 "provider": "OpenRouter", "selected": True,
                             }]},
                         }})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenRouterClient("key")
    assert client.supports_input_modality("openrouter/free", "image") is True
    assert client.supports_tools("openrouter/free") is True
    result = client.chat("openrouter/free", [{"role": "user", "content": "x"}])
    assert result.model == "dots-studio/dots-3-note-preview:free"
    assert result.usage["beatforge_model_route"] == {
        "requested_model": "openrouter/free",
        "selected_model": "dots-studio/dots-3-note-preview:free",
        "selected_provider": "OpenRouter",
        "strategy": "free-models",
        "is_byok": False,
    }


def test_openrouter_rejects_priced_free_router_catalog_entry(monkeypatch):
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps({"data": [{"id": "openrouter/free",
                                          "pricing": {"prompt": "0", "completion": "0.01"}}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="priced model"):
        OpenRouterClient("key").chat("openrouter/free", [{"role": "user", "content": "x"}])


def test_openrouter_streams_visible_content_and_usage(monkeypatch):
    payloads = []
    partials = []

    class Response:
        def __init__(self, data):
            self.data = data
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps(self.data).encode()
        def __iter__(self):
            return iter(self.data)

    def sse(value):
        return b"data: " + json.dumps(value).encode() + b"\n\n"

    def fake_urlopen(req, **kwargs):
        if req.data is None:
            return Response({"data": [{"id": "demo/model:free",
                                       "pricing": {"prompt": "0", "completion": "0"}}]})
        payloads.append(json.loads(req.data.decode()))
        return Response([
            sse({"model": "demo/model:free", "choices": [{"delta": {
                "content": "map.note(", "reasoning": "not part of the visible response"}}]}),
            sse({"choices": [{"delta": {"content": "1, 1, 0, 0, 1)"}}]}),
            sse({"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                           "total_tokens": 18, "cost": 0}}),
            b"data: [DONE]\n\n",
        ])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenRouterClient("key")
    client.set_progress_callback(partials.append)
    result = client.chat("demo/model:free", [{"role": "user", "content": "map"}])
    assert result.content == "map.note(1, 1, 0, 0, 1)"
    assert result.usage["total_tokens"] == 18
    assert partials[0] == ""
    assert partials[-1] == result.content
    assert all("reasoning" not in partial for partial in partials)
    assert payloads[0]["stream"] is True
    assert payloads[0]["stream_options"] == {"include_usage": True}
    assert "max_tokens" not in payloads[0]


def test_openrouter_stream_records_selected_free_router_endpoint(monkeypatch):
    class Response:
        def __init__(self, data):
            self.data = data
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps(self.data).encode()
        def __iter__(self):
            return iter(self.data)

    def sse(value):
        return b"data: " + json.dumps(value).encode() + b"\n\n"

    def fake_urlopen(req, **kwargs):
        if req.data is None:
            return Response({"data": [{"id": "openrouter/free",
                                       "pricing": {"prompt": "0", "completion": "0"}}]})
        assert {key.lower(): value for key, value in req.header_items()}[
            "x-openrouter-metadata"] == "enabled"
        return Response([
            sse({"model": "openrouter/free", "choices": [{"delta": {"content": "map.note()"}}]}),
            sse({"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": {"cost": 0, "is_byok": False},
                 "openrouter_metadata": {"is_byok": False,
                    "endpoints": {"available": [{"model": "google/gemma-4-31b-it:free",
                        "provider": "Google AI Studio", "selected": True}]}}}),
            b"data: [DONE]\n\n",
        ])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenRouterClient("key")
    client.set_progress_callback(lambda _partial: None)
    result = client.chat("openrouter/free", [{"role": "user", "content": "map"}])

    assert result.model == "google/gemma-4-31b-it:free"
    assert result.usage["beatforge_model_route"]["selected_provider"] == "Google AI Studio"
    assert result.usage["beatforge_model_route"]["is_byok"] is False


def test_openrouter_streams_tool_call_deltas(monkeypatch):
    class Response:
        def __init__(self, data):
            self.data = data
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps(self.data).encode()
        def __iter__(self):
            return iter(self.data)

    def sse(value):
        return b"data: " + json.dumps(value).encode() + b"\n\n"

    def fake_urlopen(req, **kwargs):
        if req.data is None:
            return Response({"data": [{"id": "demo/model:free",
                                       "pricing": {"prompt": "0", "completion": "0"},
                                       "supported_parameters": ["tools"]}]})
        return Response([
            sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1",
                   "type": "function", "function": {"name": "search", "arguments": "{\"q\":"}}]}}]}),
            sse({"choices": [{"delta": {"tool_calls": [{"index": 0,
                   "function": {"arguments": "\"moon\"}"}}]}, "finish_reason": "tool_calls"}]}),
            b"data: [DONE]\n\n",
        ])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenRouterClient("key")
    client.set_progress_callback(lambda _partial: None)
    result = client.chat(
        "demo/model:free", [{"role": "user", "content": "search"}], tools=[{}])
    assert result.content == ""
    assert result.tool_calls[0]["id"] == "call-1"
    assert result.tool_calls[0]["function"] == {
        "name": "search", "arguments": '{"q":"moon"}',
    }


def test_openrouter_never_sends_token_caps(monkeypatch):
    payloads = []

    class Response:
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(req, **kwargs):
        if req.data is None:
            return Response({"data": [{"id": "demo/model:free",
                                       "pricing": {"prompt": "0", "completion": "0"}}]})
        payloads.append(json.loads(req.data.decode()))
        return Response({"choices": [{"message": {"content": "map.note(...)"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(TypeError):
        OpenRouterClient("key").chat("demo/model:free", [{"role": "user", "content": "x"}],
                                      max_tokens=2048)
    OpenRouterClient("key").chat("demo/model:free", [{"role": "user", "content": "x"}],
                                   reasoning_effort="high")
    assert "max_tokens" not in payloads[0]
    assert "reasoning" not in payloads[0] or payloads[0]["reasoning"] == {"effort": "high"}


def test_openrouter_parses_and_transports_tool_calls(monkeypatch):
    captured = []
    tool = {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}

    class Response:
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(req, **kwargs):
        if req.data is None:
            return Response({"data": [{"id": "demo/model:free",
                                       "pricing": {"prompt": "0", "completion": "0"},
                                       "supported_parameters": ["tools"]}]})
        captured.append(json.loads(req.data.decode()))
        return Response({"model": "demo/model:free", "choices": [{"message": {
            "role": "assistant", "content": None, "tool_calls": [{"id": "call-1",
                "type": "function", "function": {"name": "search", "arguments": "{}"}}]}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = OpenRouterClient("key").chat(
        "demo/model:free", [{"role": "user", "content": "search"}],
        tools=[tool], tool_choice="auto", parallel_tool_calls=False,
    )
    assert result.content == ""
    assert result.tool_calls[0]["id"] == "call-1"
    assert captured[0]["tools"] == [tool]
    assert captured[0]["parallel_tool_calls"] is False


def test_openrouter_rejects_tools_when_free_model_catalog_does_not_advertise_them(monkeypatch):
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps({"data": [{"id": "demo/model:free",
                                         "pricing": {"prompt": "0", "completion": "0"},
                                         "supported_parameters": []}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    client = OpenRouterClient("key")
    with pytest.raises(ValueError, match="tool support"):
        client.chat("demo/model:free", [{"role": "user", "content": "search"}], tools=[{}])


def test_openrouter_transports_reasoning_effort(monkeypatch):
    captured = {}

    class Response:
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(req, **kwargs):
        if req.data is None:
            return Response({"data": [{"id": "demo/model:free",
                                       "pricing": {"prompt": "0", "completion": "0"}}]})
        captured["payload"] = json.loads(req.data.decode())
        return Response({"choices": [{"message": {"content": "map.note(...)"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    OpenRouterClient("key").chat("demo/model:free", [{"role": "user", "content": "x"}],
                                   reasoning_effort="xhigh")
    assert captured["payload"]["reasoning"] == {"effort": "xhigh"}


def test_openrouter_rejects_reasoning_token_cap_argument():
    with pytest.raises(TypeError):
        OpenRouterClient("key").chat("demo/model:free", [{"role": "user", "content": "x"}],
                                      reasoning_max_tokens=2048)


def test_openrouter_chat_rejects_priced_suffix_before_post(monkeypatch):
    calls = []

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps({"data": [{"id": "demo/fake:free",
                                         "pricing": {"prompt": "0.01", "completion": "0"}}]}).encode()

    def fake_urlopen(req, **kwargs):
        calls.append(req.data)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ValueError, match="priced model"):
        OpenRouterClient("key").chat("demo/fake:free", [{"role": "user", "content": "x"}])
    assert calls == [None]  # catalog GET only; generation POST never occurs


def test_openrouter_empty_model_output_reports_finish_metadata(monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(req, **kwargs):
        if req.data is None:
            return Response({"data": [{"id": "demo/model:free",
                                       "pricing": {"prompt": "0", "completion": "0"}}]})
        return Response({"model": "demo/actual-model:free",
                         "user_id": "must-not-be-persisted",
                         "choices": [{"finish_reason": "error",
                                       "native_finish_reason": "ProviderError",
                                       "message": {"content": None,
                                                   "reasoning": "private reasoning text"}}],
                         "openrouter_metadata": {"is_byok": False,
                            "endpoints": {"available": [{"selected": True,
                                "model": "demo/actual-model:free",
                                "provider": "Demo Provider"}]}},
                         "usage": {"prompt_tokens": 8, "completion_tokens": 13,
                                   "total_tokens": 21}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="finish_reason='error'.*reasoning_returned=True") as exc:
        OpenRouterClient("key").chat("demo/model:free", [{"role": "user", "content": "x"}])
    assert exc.value.usage == {
        "prompt_tokens": 8, "completion_tokens": 13, "total_tokens": 21,
    }
    assert exc.value.finish_reason == "error"
    assert exc.value.diagnostics["selected_model"] == "demo/actual-model:free"
    assert exc.value.diagnostics["choice"] == {
        "finish_reason": "error",
        "native_finish_reason": "ProviderError",
        "message_fields": ["content", "reasoning"],
    }
    assert exc.value.diagnostics["route"] == {
        "is_byok": False,
        "selected_model": "demo/actual-model:free",
        "selected_provider": "Demo Provider",
    }
    assert "private reasoning text" not in repr(exc.value.diagnostics)
    assert "must-not-be-persisted" not in repr(exc.value.diagnostics)


def test_openrouter_no_choices_keeps_provider_error_details(monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(req, **kwargs):
        if req.data is None:
            return Response({"data": [{"id": "demo/model:free",
                                       "pricing": {"prompt": "0", "completion": "0"}}]})
        return Response({"error": {"code": 503, "message": "provider unavailable",
                                    "metadata": {"provider_name": "Nvidia",
                                                 "raw": "{\"error\": {\"message\": \"temporary capacity error\"}}"}},
                         "user_id": "must-not-be-persisted"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="code=503.*provider='Nvidia'.*provider unavailable") as exc:
        OpenRouterClient("key").chat("demo/model:free", [{"role": "user", "content": "x"}])
    assert exc.value.diagnostics["error"] == {
        "code": "503",
        "message": "provider unavailable",
        "metadata": {"provider_name": "Nvidia",
                      "provider_message": "temporary capacity error"},
    }
    assert "must-not-be-persisted" not in repr(exc.value.diagnostics)


def test_openrouter_modality_lookup_requires_catalogued_zero_price_free_model():
    free_audio = {"id": "demo/audio:free", "pricing": {"prompt": "0", "completion": "0"},
                  "architecture": {"input_modalities": ["text", "audio"]},
                  "supported_parameters": []}
    client = OpenRouterClient("key")
    client.catalog = lambda: [free_audio]
    assert client.supports_input_modality("demo/audio:free", "audio") is True
    assert client.supports_input_modality("demo/audio:free", "image") is False

    priced = OpenRouterClient("key")
    priced.catalog = lambda: [{**free_audio, "pricing": {"prompt": "0.01", "completion": "0"}}]
    with pytest.raises(ValueError, match="priced model"):
        priced.supports_input_modality("demo/audio:free", "audio")


def test_local_client_requires_loopback_and_allows_missing_key(monkeypatch):
    with pytest.raises(ValueError, match="loopback"):
        LocalOpenAIClient(base_url="https://example.test/v1")

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps({"model": "gemma", "choices": [{"message": {"content": "map.note(...)"}}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    result = LocalOpenAIClient().chat("gemma", [{"role": "user", "content": "x"}])
    assert result.model == "gemma"


def test_local_client_rejects_redirect_escape(monkeypatch):
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def geturl(self):
            return "https://example.test/v1/chat/completions"
        def read(self):
            return b"{}"

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError, match="escaped loopback"):
        LocalOpenAIClient().chat("gemma", [{"role": "user", "content": "x"}])

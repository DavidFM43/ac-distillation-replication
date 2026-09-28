"""Offline tests: VllmTokenCompleter against a fake vLLM /v1/completions (httpx.MockTransport)."""

import asyncio
import json

import httpx
import pytest
import tinker

from sd import vllm_policy
from sd.vllm_policy import VllmTokenCompleter, find_endpoint

IM_END = 248046


def completion(tokens, logprobs, finish_reason="stop"):
    return {"choices": [{
        "text": "…",
        "finish_reason": finish_reason,
        "logprobs": {"tokens": [f"token_id:{t}" for t in tokens], "token_logprobs": logprobs},
    }]}


def make_completer(handler, **kwargs):
    requests = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    completer = VllmTokenCompleter(
        "http://vllm.test/v1", "secret", "Qwen/Qwen3.5-9B", max_tokens=16384,
        transport=httpx.MockTransport(record), **kwargs,
    )
    return completer, requests


def call(completer, prompt=(1, 2, 3), stop=(IM_END,), max_tokens=None):
    return asyncio.run(completer(tinker.ModelInput.from_ints(list(prompt)), list(stop), max_tokens=max_tokens))


def test_sends_exact_token_ids_and_parses_sampled_tokens():
    completer, requests = make_completer(lambda r: httpx.Response(200, json=completion([7, 8, IM_END], [-0.1, -0.2, -0.3])))
    result = call(completer, max_tokens=100)

    body = json.loads(requests[0].content)
    assert requests[0].url == "http://vllm.test/v1/completions"
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert body["prompt"] == [1, 2, 3]
    assert body["stop_token_ids"] == [IM_END]
    assert body["max_tokens"] == 100  # per-call cap wins over the completer's 16384
    assert body["logprobs"] == 0 and body["return_tokens_as_token_ids"] is True
    assert result.tokens == [7, 8, IM_END]
    assert result.logprobs == [-0.1, -0.2, -0.3]
    assert result.stop_reason == "stop"


def test_length_finish_is_reported():
    completer, _ = make_completer(lambda r: httpx.Response(200, json=completion([7, 8], [-0.1, -0.2], "length")))
    assert call(completer).stop_reason == "length"


def test_stop_without_stop_token_is_an_error():
    completer, _ = make_completer(lambda r: httpx.Response(200, json=completion([7, 8], [-0.1, -0.2])))
    with pytest.raises(RuntimeError, match="without a stop token"):
        call(completer)


def test_retries_transient_server_errors(monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr(vllm_policy.asyncio, "sleep", no_sleep)
    responses = iter([httpx.Response(503), httpx.Response(200, json=completion([IM_END], [-0.1]))])
    completer, requests = make_completer(lambda r: next(responses))
    assert call(completer).tokens == [IM_END]
    assert len(requests) == 2


def test_endpoint_from_env_then_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="No vLLM endpoint"):
        find_endpoint()
    (tmp_path / ".vllm_endpoint.json").write_text(json.dumps({"base_url": "http://127.0.0.1:8000/v1/", "api_key": "k"}))
    assert find_endpoint() == ("http://127.0.0.1:8000/v1", "k")
    monkeypatch.setenv("VLLM_BASE_URL", "http://other:9000/v1")
    monkeypatch.setenv("VLLM_API_KEY", "k2")
    assert find_endpoint() == ("http://other:9000/v1", "k2")

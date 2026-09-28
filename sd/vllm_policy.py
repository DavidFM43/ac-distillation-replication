"""Qwen3.5 on a self-hosted vLLM server as a token-level policy.

The rollout script loads it by import path: `--policy sd.vllm_policy:make_vllm_policy`.

The server is found from VLLM_BASE_URL / VLLM_API_KEY, or else from `.vllm_endpoint.json`
(written by `infra/vllm_host.py up`). Prompts go to vLLM's /v1/completions as token ids, so
the renderer's exact tokens reach the model, and the sampled token ids and their logprobs
come back unchanged.
"""

import asyncio
import json
import os
from pathlib import Path

import httpx
import tinker
from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs

ENDPOINT_FILE = Path(".vllm_endpoint.json")
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def find_endpoint() -> tuple[str, str]:
    """(base_url, api_key) from the environment, else from .vllm_endpoint.json."""
    if base_url := os.environ.get("VLLM_BASE_URL"):
        return base_url.rstrip("/"), os.environ.get("VLLM_API_KEY", "")
    if ENDPOINT_FILE.exists():
        endpoint = json.loads(ENDPOINT_FILE.read_text())
        return endpoint["base_url"].rstrip("/"), endpoint["api_key"]
    raise RuntimeError(f"No vLLM endpoint: set VLLM_BASE_URL (and VLLM_API_KEY) or create {ENDPOINT_FILE}")


class VllmTokenCompleter(TokenCompleter):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        max_tokens: int,
        temperature: float = 1.0,
        max_retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,  # for tests
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        # No overall timeout: a 16k-token turn can take minutes on a small GPU.
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=httpx.Timeout(None, connect=30.0),
            transport=transport,
        )

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition, *, max_tokens: int | None = None
    ) -> TokensWithLogprobs:
        body = {
            "model": self.model_name,
            "prompt": model_input.to_ints(),
            "max_tokens": min(self.max_tokens, max_tokens) if max_tokens is not None else self.max_tokens,
            "temperature": self.temperature,
            "logprobs": 0,  # logprob of each sampled token
            "return_tokens_as_token_ids": True,  # tokens come back as "token_id:<id>"
        }
        stop_ids = [s for s in stop if isinstance(s, int)]
        body |= {"stop_token_ids": stop_ids} if stop_ids else {"stop": list(stop)}

        choice = (await self._post_completion(body))["choices"][0]
        tokens = [int(t.removeprefix("token_id:")) for t in choice["logprobs"]["tokens"]]
        logprobs = choice["logprobs"]["token_logprobs"]
        stop_reason = "length" if choice["finish_reason"] == "length" else "stop"
        if stop_reason == "stop" and stop_ids and (not tokens or tokens[-1] not in stop_ids):
            # The renderer's parser expects the end-of-turn token, as Tinker returns it.
            raise RuntimeError(f"vLLM stopped without a stop token (last token {tokens[-1:]}, stop {stop_ids})")
        return TokensWithLogprobs(tokens=tokens, maybe_logprobs=logprobs, stop_reason=stop_reason)

    async def _post_completion(self, body: dict) -> dict:
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.post("/completions", json=body)
                if response.status_code not in RETRYABLE_STATUS or attempt == self.max_retries:
                    response.raise_for_status()
                    return response.json()
            except httpx.TransportError:  # e.g. the SSH tunnel dropped and came back
                if attempt == self.max_retries:
                    raise
            await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")


def make_vllm_policy(model_name: str, max_tokens: int, temperature: float = 1.0) -> VllmTokenCompleter:
    """Policy factory for `--policy sd.vllm_policy:make_vllm_policy`.

    `model_name` must be the name vLLM serves, e.g. Qwen/Qwen3.5-9B.
    """
    base_url, api_key = find_endpoint()
    return VllmTokenCompleter(base_url, api_key, model_name, max_tokens, temperature)

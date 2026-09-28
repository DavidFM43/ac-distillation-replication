"""Checks for a running self-hosted vLLM endpoint (reads .vllm_endpoint.json). Standard library only.

    uv run python infra/check_endpoint.py

1. auth: /v1/models rejects requests without the API key and accepts them with it
2. sampling from exact token ids (tokenized by the server's /tokenize), with logprobs
3. prefix cache: a fresh ~2k-token prompt sent twice; the repeat should hit the cache for every
   full cache block (vLLM uses 528-token blocks for Qwen3.5's hybrid layers)
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT_FILE = Path(".vllm_endpoint.json")
FILLER = (
    "The repository contains a Python package with a bug in how it parses configuration files. "
    "Read the relevant modules, reproduce the failure, and propose a minimal fix. "
)


def request(url: str, api_key: str | None, body: dict | None = None) -> tuple[int, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=300) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def tokenize(root: str, api_key: str, model: str, text: str) -> list[int]:
    status, body = request(f"{root}/tokenize", api_key, {"model": model, "prompt": text, "add_special_tokens": False})
    if status != 200:
        raise SystemExit(f"/tokenize failed ({status}): {body[:300]}")
    return json.loads(body)["tokens"]


def prefix_cache_counters(root: str) -> dict[str, float]:
    _, text = request(f"{root}/metrics", None)
    counters: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("vllm:prefix_cache_") and "_total" in line:
            name, value = line.rsplit(" ", 1)
            name = name.split("{")[0]
            counters[name] = counters.get(name, 0.0) + float(value)
    return counters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-block", type=int, default=528, help="Expected prefix-cache block size in tokens")
    args = parser.parse_args()

    endpoint = json.loads(ENDPOINT_FILE.read_text())
    base, key, model = endpoint["base_url"], endpoint["api_key"], endpoint["model"]
    root = base.removesuffix("/v1")

    # 1. auth
    no_key, _ = request(f"{base}/models", None)
    with_key, _ = request(f"{base}/models", key)
    print(f"1. auth: without key -> {no_key}, with key -> {with_key}"
          f" -> {'ok' if (no_key, with_key) == (401, 200) else 'UNEXPECTED'}")

    # 2. sampling from exact token ids
    prompt = tokenize(root, key, model, "Write one sentence about why unit tests matter.")
    t0 = time.time()
    status, body = request(f"{base}/completions", key, {
        "model": model, "prompt": prompt, "max_tokens": 128, "temperature": 1.0, "logprobs": 0,
    })
    if status != 200:
        raise SystemExit(f"2. sampling failed ({status}): {body[:300]}")
    choice = json.loads(body)["choices"][0]
    n_out = len(choice["logprobs"]["token_logprobs"])
    elapsed = time.time() - t0
    print(f"2. sampling: {n_out} tokens with logprobs in {elapsed:.1f}s ({n_out / elapsed:.0f} tok/s incl. tunnel)")

    # 3. prefix cache: a unique marker first, so the first request can't hit anything cached earlier
    text = f"[run {time.time_ns()}] " + FILLER * 60
    long_prompt = tokenize(root, key, model, text)
    before = prefix_cache_counters(root)
    for _ in range(2):
        request(f"{base}/completions", key, {"model": model, "prompt": long_prompt, "max_tokens": 1, "temperature": 0})
    after = prefix_cache_counters(root)
    queries = after.get("vllm:prefix_cache_queries_total", 0) - before.get("vllm:prefix_cache_queries_total", 0)
    hits = after.get("vllm:prefix_cache_hits_total", 0) - before.get("vllm:prefix_cache_hits_total", 0)
    expected = (len(long_prompt) // args.cache_block) * args.cache_block
    print(f"3. prefix cache: {len(long_prompt)}-token prompt twice -> {hits:.0f} hits of {queries:.0f} queried tokens"
          f" (expected ~{expected} from full {args.cache_block}-token blocks on the repeat)")


if __name__ == "__main__":
    main()

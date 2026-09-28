"""Plumbing checks for a self-hosted vLLM endpoint (reads .vllm_endpoint.json).

    uv run python infra/check_endpoint.py --trajectories runs/baseline/trajectories.jsonl

1. auth: /v1/models rejects requests without the key, accepts them with it
2. renderer parity: re-rendering a stored trajectory's first prompt gives Tinker's token count
3. logprob parity: vLLM's logprobs for the tokens Tinker sampled vs Tinker's stored logprobs
4. sampling from exact token ids, with logprobs
5. prefix cache: the same long prompt twice, cache-hit counters before/after
"""

import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

import typer

from sd.renderer import get_agent_renderer

ENDPOINT_FILE = Path(".vllm_endpoint.json")


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


def prefix_cache_counters(root: str) -> dict[str, float]:
    _, text = request(f"{root}/metrics", None)
    counters = {}
    for line in text.splitlines():
        if line.startswith("vllm:prefix_cache_") and not line.startswith("#"):
            name, value = line.rsplit(" ", 1)
            counters[name.split("{")[0]] = counters.get(name.split("{")[0], 0.0) + float(value)
    return counters


def main(trajectories: Path = Path("runs/baseline/trajectories.jsonl"), instance: str = "") -> None:
    endpoint = json.loads(ENDPOINT_FILE.read_text())
    base, key, model = endpoint["base_url"], endpoint["api_key"], endpoint["model"]
    root = base.removesuffix("/v1")

    # 1. auth
    no_key, _ = request(f"{base}/models", None)
    with_key, _ = request(f"{base}/models", key)
    print(f"1. auth: without key -> {no_key}, with key -> {with_key}")

    # 2. renderer parity on a stored trajectory's first turn (system + user messages only)
    records = [json.loads(line) for line in trajectories.read_text().splitlines()]
    record = next(r for r in records if r["instance_id"] == instance) if instance else min(records, key=lambda r: r["n_turns"])
    renderer = get_agent_renderer(record["model_name"])
    prompt = renderer.build_generation_prompt(record["messages"][:2]).to_ints()
    turn = record["turns"][0]
    print(f"2. renderer parity ({record['instance_id']}): re-rendered {len(prompt)} tokens, Tinker saw {turn['prompt_len']}"
          f" -> {'match' if len(prompt) == turn['prompt_len'] else 'MISMATCH'}")

    # 3. score Tinker's sampled tokens under vLLM
    completion = turn["tokens"]
    status, body = request(f"{base}/completions", key, {
        "model": model, "prompt": prompt + completion, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0,
    })
    if status != 200:
        raise typer.Exit(f"3. scoring request failed ({status}): {body[:300]}")
    scored = json.loads(body)["choices"][0]["prompt_logprobs"][len(prompt):]
    vllm_lp = [entry[str(tok)]["logprob"] for entry, tok in zip(scored, completion)]
    diffs = [abs(a - b) for a, b in zip(vllm_lp, turn["logprobs"])]
    print(f"3. logprob parity on {len(diffs)} sampled tokens: mean |vLLM - Tinker| = {statistics.mean(diffs):.4f}, "
          f"median = {statistics.median(diffs):.4f}, max = {max(diffs):.3f}")

    # 4. sampling from exact token ids
    t0 = time.time()
    status, body = request(f"{base}/completions", key, {
        "model": model, "prompt": prompt, "max_tokens": 128, "temperature": 1.0, "logprobs": 0,
    })
    choice = json.loads(body)["choices"][0]
    n_out = len(choice["logprobs"]["token_logprobs"])
    print(f"4. sampling: {status}, {n_out} tokens in {time.time() - t0:.1f}s, first words: {choice['text'][:80]!r}")

    # 5. prefix cache: same long prompt twice
    long_prompt = prompt + completion
    before = prefix_cache_counters(root)
    for _ in range(2):
        request(f"{base}/completions", key, {"model": model, "prompt": long_prompt, "max_tokens": 1, "temperature": 0})
    after = prefix_cache_counters(root)
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    print(f"5. prefix cache over 2 requests of {len(long_prompt)} tokens: {delta}")


if __name__ == "__main__":
    typer.run(main)

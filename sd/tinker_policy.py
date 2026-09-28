"""Qwen3.5 on Tinker as a token-level policy, plus Tinker cost estimates for rollout records.

A policy factory takes `(model_name, max_tokens, temperature)` and returns a tinker_cookbook
`TokenCompleter`; the rollout script loads it by import path, e.g.
`--policy sd.tinker_policy:make_tinker_policy`.

    uv run python -m sd.tinker_policy runs/baseline/trajectories.jsonl   # estimated Tinker cost
"""

import json
from pathlib import Path

import tinker
import typer
from tinker_cookbook.completers import TinkerTokenCompleter

# USD per million tokens (Tinker docs, Sept 2026).
PRICES = {
    "Qwen/Qwen3.5-9B": {"prefill": 0.66, "cached": 0.132, "sample": 1.995},
    "Qwen/Qwen3.5-4B": {"prefill": 0.33, "cached": 0.066, "sample": 1.005},
}


def make_tinker_policy(model_name: str, max_tokens: int, temperature: float = 1.0) -> TinkerTokenCompleter:
    """Temperature 1.0 (no top-p/top-k) keeps sampled logprobs equal to the model's own distribution."""
    sampling_client = tinker.ServiceClient().create_sampling_client(base_model=model_name)
    return TinkerTokenCompleter(sampling_client, max_tokens=max_tokens, temperature=temperature)


def estimate_cost(model_name: str, turns: list[dict]) -> float:
    """Tinker cost of one trajectory from its per-turn `prompt_len` and sampled `tokens`.

    Assumes the previous turn's prompt + completion hit the prefix cache.
    """
    price = PRICES[model_name]
    cost, prev_total = 0.0, 0
    for turn in turns:
        prompt, completion = turn["prompt_len"], len(turn["tokens"])
        cached = min(prev_total, prompt)
        cost += (prompt - cached) * price["prefill"] + cached * price["cached"] + completion * price["sample"]
        prev_total = prompt + completion
    return cost / 1e6


def main(trajectories: Path) -> None:
    """Print the estimated Tinker cost of a rollout JSONL."""
    records = [json.loads(line) for line in trajectories.read_text().splitlines()]
    costs = [estimate_cost(r["model_name"], r["turns"]) for r in records]
    print(f"{len(records)} trajectories: estimated Tinker cost ${sum(costs):.2f} "
          f"(mean ${sum(costs) / len(costs):.3f}, max ${max(costs):.3f})")


if __name__ == "__main__":
    typer.run(main)

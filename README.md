# sd-swebench

Replicating Applied Compute's Qwen3.5 "submit before turn 40" self-distillation result
(Sam Denton, "Bringing Continual Learning into Enterprises", AI Engineer).

## Qwen3.5 on Tinker

`sd/tinker_policy.py` provides the Tinker backend for rollouts:

- `make_tinker_policy(model_name, max_tokens, temperature)` returns a tinker_cookbook
  `TokenCompleter` that samples on Tinker. It takes prompt token ids and returns the sampled
  token ids and their logprobs. With temperature 1.0 and no top-p/top-k, those logprobs are the
  model's own distribution. The rollout script loads it by import path:
  `--policy sd.tinker_policy:make_tinker_policy`.
- `estimate_cost(model_name, turns)` estimates a trajectory's Tinker cost from each turn's
  `prompt_len` and sampled `tokens`, assuming the previous turn hits the prefix cache.

```bash
uv sync
export TINKER_API_KEY=...   # or put it in a gitignored .env
uv run python -m sd.tinker_policy runs/baseline/trajectories.jsonl   # estimated Tinker cost of a run
uv run pytest
```

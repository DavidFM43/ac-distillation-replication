# sd-swebench

Replicating Applied Compute's Qwen3.5 "submit before turn 40" self-distillation result
(Sam Denton, "Bringing Continual Learning into Enterprises", AI Engineer).

## Qwen3.5 on a self-hosted vLLM server

`sd/vllm_policy.py` provides the vLLM backend for rollouts. `make_vllm_policy(model_name,
max_tokens, temperature)` returns a tinker_cookbook `TokenCompleter` that sends prompt token
ids to vLLM's `/v1/completions` and returns the sampled token ids with their logprobs. The
rollout script loads it by import path: `--policy sd.vllm_policy:make_vllm_policy`.

- **Endpoint:** `VLLM_BASE_URL` and `VLLM_API_KEY` if set, otherwise `.vllm_endpoint.json`
  (written by `infra/vllm_host.py up`).
- **Token-exact:** prompts are sent as token ids, so the renderer's exact tokens reach the
  model. Sampled tokens come back as ids (`return_tokens_as_token_ids`), not re-tokenized text.
- **Stop token:** the renderer's end-of-turn token is passed as `stop_token_ids`. If vLLM
  reports a normal stop but the last token isn't a stop token, the call fails rather than
  storing a malformed turn.
- **Retries:** transient errors (tunnel drops, 429/5xx) are retried with backoff. There's no
  overall request timeout, since a long turn can take minutes on a small GPU.
- **Model name:** `--model-name` must match the model vLLM serves, e.g. `Qwen/Qwen3.5-9B`.

```bash
uv run python -m sd.harbor_rollouts --policy sd.vllm_policy:make_vllm_policy \
    --n-tasks 10 --output runs/vllm/trajectories.jsonl
uv run pytest
```

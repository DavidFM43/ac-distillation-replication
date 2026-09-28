# sd-swebench

Replicating Applied Compute's Qwen3.5 "submit before turn 40" self-distillation result
(Sam Denton, "Bringing Continual Learning into Enterprises", AI Engineer).

## Self-hosted vLLM (RunPod or AWS via SkyPilot)

`infra/vllm_host.py` runs vLLM on a rented GPU and reaches it over an SSH tunnel.

- `deploy/qwen35_9b.yaml` is a deployment: the pinned engine (`engine: vllm==0.30.0`,
  pip-installed into its own venv on the node), its env vars, and the `vllm serve` settings.
  It says nothing about hardware.
- `--gpus` picks the hardware, and accepts an any-of list (SkyPilot takes the cheapest with
  capacity). The tensor-parallel size comes from the node's GPU count. `--backend` is
  `runpod` or `aws`.
- The server listens only on the node's localhost. `up` opens `ssh -L` and writes
  `.vllm_endpoint.json` (gitignored, mode 600) with the local `base_url` and a per-deployment
  API key. The key reaches the server as a SkyPilot secret.
- The cluster is deleted after `--max-minutes`, unconditionally, and also if a launch fails
  after the node exists.

```bash
uv sync --group infra
uv run --group infra sky check runpod aws
uv run --group infra python infra/vllm_host.py up --config deploy/qwen35_9b.yaml \
    --gpus "A40:1,RTXA6000:1,L40:1,RTX6000-Ada:1,L40S:1" --backend runpod --max-minutes 60
uv run --group infra python infra/vllm_host.py status
uv run --group infra python infra/vllm_host.py down
```

Notes from the first deployment (vLLM 0.30.0, A40):

- The `vllm/vllm-openai` image's `vllm serve` entrypoint swallows SkyPilot's SSH setup on
  RunPod, which is why vLLM is pip-installed instead.
- FlashInfer's sampler JIT-compiles a kernel and needs `nvcc`, which the base images lack, so
  the deployment sets `VLLM_USE_FLASHINFER_SAMPLER=0`.
- Prefix caching works for Qwen3.5's hybrid layers, in 528-token blocks.
- MTP speculative decoding is left off because of an open vLLM report of garbled tool calls
  with prefix caching on these models.

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

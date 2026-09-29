# sd-swebench

Replicating Applied Compute's Qwen3.5 "submit before turn 40" self-distillation result
(Sam Denton, "Bringing Continual Learning into Enterprises", AI Engineer).

## SWE-bench rollouts with Harbor

`sd/harbor_rollouts.py` runs a SWE-bench Verified agent through tinker_cookbook's token-level
agent loop.

- Each task (Harbor format) runs in a Modal sandbox sized from its `task.toml`.
- Tools: `bash` (runs in `/testbed` with the repo's conda env, with mini-swe-agent-style
  output) and `task_complete` (ends the episode).
- Prompt: mini-swe-agent's SWE-bench prompt (MIT), adapted to submit via `task_complete`, in
  `sd/prompts.py`.
- After every episode, whatever ended it, the task's `tests/test.sh` grades the sandbox's
  final state. So "tests pass" is measured independently of "submitted".

**The model backend is pluggable.** `--policy` names a factory by import path. The factory
takes `(model_name, max_tokens, temperature)` and returns a tinker_cookbook `TokenCompleter`:
prompt token ids in, sampled token ids and logprobs out. This module doesn't import any
backend.

**`sd/renderer.py`** converts between messages and tokens. `Qwen35AgentRenderer` matches
Qwen's official chat template in agent loops:

- It keeps `<think>` blocks in history. The cookbook default strips them; the template keeps
  them for every turn after the last real user message.
- It trims text around thinking and tool calls, like the template.

Together, each turn's prompt is exactly the previous prompt + the sampled tokens + the tool
results. Token-level teacher/student re-scoring relies on that.

Setup (once):

```bash
uv sync
uv run modal setup
uvx harbor datasets download swebench-verified@1.0 -o ~/.cache/harbor/tasks/swebench-verified-1.0
```

Run and summarize (with a policy backend installed, e.g. the Tinker one):

```bash
uv run python -m sd.harbor_rollouts --policy sd.tinker_policy:make_tinker_policy \
    --n-tasks 10 --output runs/baseline/trajectories.jsonl
uv run python -m sd.summarize runs/baseline/trajectories.jsonl --threshold 40
```

The task sample is a fixed shuffle (`--seed 0`). The model is never told about a turn limit;
`--max-turns 150` is only a safety cap.

Each JSONL line holds:

- `instance_id`, the policy and sampling config, `stop_reason` (`tool_stopped` = called
  task_complete, `completed` = replied without a tool call, `max_turns`, `context_overflow`,
  `max_tokens`, `parse_error`), and `error` if the run crashed
- `submitted`, `submit_turn`, `n_turns`, `tests_passed`, `elapsed_s`
- `messages`: the full conversation, including thinking
- `turns`: per assistant turn, `prompt_len` and the exact sampled `tokens` and `logprobs`.
  Re-render `messages` up to that turn with `Qwen35AgentRenderer` to get the prompt tokens back.

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
- vLLM's PyPI wheel is built for CUDA 13 and fails on RunPod hosts with older drivers (e.g. 570,
  CUDA 12.8), so the deployment installs vLLM's CUDA 12.9 build (`cuda: cu129`) with the
  matching PyTorch, which runs on old and new drivers alike.
- FlashInfer's sampler JIT-compiles a kernel and needs `nvcc`, which the base images lack, so
  the deployment sets `VLLM_USE_FLASHINFER_SAMPLER=0`.
- Prefix caching works for Qwen3.5's hybrid layers, in 528-token blocks.
- MTP speculative decoding is left off because of an open vLLM report of garbled tool calls
  with prefix caching on these models.

**CPU KV-cache offloading (untested):** `vllm_host.py up --cpu-offload auto` gives vLLM a RAM tier
for evicted KV-cache blocks (`--kv-offloading-size`, native backend). With `auto`, the size is the
node's available memory, capped by the container's cgroup limit, minus 12 GiB. It's off by default
until verified on a pod: a hybrid-GDN model can silently write to RAM without ever reading back
(vllm-project/vllm#58653, triggered by an explicit `--block-size`, which this deployment doesn't set).

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

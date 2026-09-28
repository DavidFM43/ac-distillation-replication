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

```bash
uv run pytest
```

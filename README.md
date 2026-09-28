# sd-swebench

Replicating Applied Compute's Qwen3.5 "submit before turn 40" self-distillation result
(Sam Denton, "Bringing Continual Learning into Enterprises", AI Engineer).

Step 1 is reproducing the problem: how often does Qwen3.5-9B, run normally on SWE-bench,
fail to submit (or submit late)?

## Layout

- `sd/harbor_rollouts.py`: rollouts with tinker_cookbook's agent loop. Qwen3.5 is sampled on
  Tinker, and each SWE-bench Verified task (Harbor format) runs in a Modal sandbox. Tools are
  `bash` and `task_complete`. After each episode, the task's tests run on the sandbox's final
  state, whatever ended the episode.
- `sd/renderer.py`: Qwen3.5 renderer that matches Qwen's official chat template in agent loops
  (keeps thinking in history, trims text like the template).
- `sd/prompts.py`: mini-swe-agent's SWE-bench prompt (MIT), adapted to submit via `task_complete`.
- `sd/summarize.py`: task complete rate, test pass rate, submit turn distribution, with 95% intervals.

## Setup (once)

```bash
uv sync
export TINKER_API_KEY=...
uv run modal setup
uvx harbor datasets download swebench-verified@1.0 -o ~/.cache/harbor/tasks/swebench-verified-1.0
```

## Baseline rollouts (10 tasks)

```bash
uv run python -m sd.harbor_rollouts --n-tasks 10 --output runs/baseline/trajectories.jsonl
uv run python -m sd.summarize runs/baseline/trajectories.jsonl --threshold 40
```

The task sample is a fixed shuffle (`--seed 0`), so the same 10 tasks are used every time.
The JSONL is appended to as tasks finish.

## Trajectory record (one JSONL line)

- `instance_id`, sampling config (`model_name`, `renderer`, `temperature`, `max_tokens`, ...)
- `stop_reason` (`tool_stopped` = called task_complete, `completed` = replied without a tool call,
  `max_turns`, `context_overflow`, `max_tokens`, `parse_error`), plus `error` if the run crashed
- `submitted`, `submit_turn`, `n_turns`, `tests_passed`, `cost_estimate`, `elapsed_s`
- `messages`: the full conversation, including thinking
- `turns`: per assistant turn, `prompt_len` and the exact sampled `tokens` and `logprobs`.
  Prompt tokens aren't stored; re-render the messages up to that turn with `Qwen35AgentRenderer`
  to get them back.

## Notes

- Temperature 1.0 with no top-p/top-k, so stored logprobs are the model's own distribution.
- Costs are estimates from token counts and Tinker's published prices, assuming the previous
  turn hits the prefix cache.
- The model is never told about a turn limit. `--max-turns 150` is only a safety cap.

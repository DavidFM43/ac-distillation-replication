"""Live progress for a rollout run: a status.json for the HTML dashboard, and Prometheus metrics.

Per-turn numbers come from `TrackingPolicy`, which wraps the run's policy per task, so the
cookbook's rollout loop is untouched. Token accounting matches the Tinker bill: tokens that were
part of the previous turn's prompt or output count as cached input; the rest is new input.
"""

import json
import os
import time
from pathlib import Path

import tinker
from prometheus_client import Counter, Gauge, start_http_server
from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs

TASKS = Gauge("sd_rollout_tasks", "Tasks by state", ["state"])
TURNS = Counter("sd_rollout_turns_total", "Assistant turns sampled")
TOKENS = Counter("sd_rollout_tokens_total", "Tokens by kind", ["kind"])  # input_new, input_cached, output
FINISHED = Counter("sd_rollout_finished_total", "Finished tasks by outcome", ["outcome"])

STATES = ("queued", "running", "done", "error")


class RunStatus:
    def __init__(self, path: Path, config: dict, instance_ids: list[str], metrics_port: int = 0):
        self.path = path
        self.started_at = time.time()
        self.config = config
        self.tasks = {iid: {"instance_id": iid, "state": "queued"} for iid in instance_ids}
        if metrics_port:
            start_http_server(metrics_port)
        self.write()

    def start(self, iid: str) -> None:
        self.tasks[iid] |= {
            "state": "running", "started_at": time.time(), "turns": 0,
            "input_new": 0, "input_cached": 0, "output": 0, "prompt_len": 0, "_prev_total": 0,
        }
        self.write()

    def on_turn(self, iid: str, prompt_len: int, output_len: int) -> None:
        task = self.tasks[iid]
        cached = min(task["_prev_total"], prompt_len)
        task["turns"] += 1
        task["input_cached"] += cached
        task["input_new"] += prompt_len - cached
        task["output"] += output_len
        task["prompt_len"] = prompt_len
        task["_prev_total"] = prompt_len + output_len
        TURNS.inc()
        TOKENS.labels("input_cached").inc(cached)
        TOKENS.labels("input_new").inc(prompt_len - cached)
        TOKENS.labels("output").inc(output_len)
        self.write()

    def finish(self, iid: str, record: dict) -> None:
        state = "error" if record["error"] else "done"
        self.tasks[iid] |= {
            "state": state, "finished_at": time.time(),
            **{k: record[k] for k in ("stop_reason", "submitted", "submit_turn", "tests_passed", "error")},
        }
        FINISHED.labels(state if state == "error" else ("tests_passed" if record["tests_passed"] else "tests_failed")).inc()
        self.write()

    def write(self) -> None:
        counts = {s: 0 for s in STATES}
        for task in self.tasks.values():
            counts[task["state"]] += 1
        for state, n in counts.items():
            TASKS.labels(state).set(n)
        tasks = [{k: v for k, v in t.items() if not k.startswith("_")} for t in self.tasks.values()]
        totals = {k: sum(t.get(k, 0) for t in tasks) for k in ("turns", "input_new", "input_cached", "output")}
        finished = [t for t in tasks if t["state"] == "done"]
        status = {
            "updated_at": time.time(),
            "started_at": self.started_at,
            "config": self.config,
            "counts": counts,
            "totals": totals
            | {"submitted": sum(bool(t.get("submitted")) for t in finished),
               "tests_passed": sum(bool(t.get("tests_passed")) for t in finished)},
            "tasks": tasks,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status))
        os.replace(tmp, self.path)  # atomic, so the dashboard never reads a half-written file


class TrackingPolicy(TokenCompleter):
    """Wraps a policy for one task and reports each sampled turn to the RunStatus."""

    def __init__(self, inner: TokenCompleter, status: RunStatus, instance_id: str):
        self.inner, self.status, self.instance_id = inner, status, instance_id

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition, *, max_tokens: int | None = None
    ) -> TokensWithLogprobs:
        result = await self.inner(model_input, stop, max_tokens=max_tokens)
        self.status.on_turn(self.instance_id, model_input.length, len(result.tokens))
        return result

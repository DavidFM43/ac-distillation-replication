"""Offline tests: task selection, resume, and live status accounting."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import tinker
from tinker_cookbook.completers import TokensWithLogprobs

from sd.harbor_rollouts import finished_ids, select_tasks
from sd.run_status import RunStatus, TrackingPolicy


def fake_tasks(n):
    return [SimpleNamespace(task_name=f"repo__repo-{i:03d}") for i in range(n)]


def test_split_is_deterministic_disjoint_and_consistent_with_all():
    tasks = fake_tasks(500)
    train = [t.task_name for t in select_tasks(tasks, 0, "train", 250, None)]
    evals = [t.task_name for t in select_tasks(tasks, 0, "eval", 250, None)]
    assert len(train) == len(evals) == 250 and not set(train) & set(evals)
    assert train == [t.task_name for t in select_tasks(list(reversed(tasks)), 0, "train", 250, None)]
    assert [t.task_name for t in select_tasks(tasks, 0, "all", 250, 10)] == train[:10]


def test_resume_skips_finished_tasks_but_retries_errors(tmp_path):
    output = tmp_path / "trajectories.jsonl"
    output.write_text("\n".join(json.dumps(r) for r in [
        {"instance_id": "a", "error": None}, {"instance_id": "b", "error": "SandboxError: boom"},
    ]) + "\n")
    assert finished_ids(output) == {"a"}
    assert finished_ids(tmp_path / "missing.jsonl") == set()


class FakePolicy:
    def __init__(self, n_out):
        self.n_out = n_out

    async def __call__(self, model_input, stop, *, max_tokens=None):
        return TokensWithLogprobs(tokens=[1] * self.n_out, maybe_logprobs=[0.0] * self.n_out)


def test_status_tracks_turns_and_token_kinds(tmp_path):
    path = tmp_path / "status.json"
    status = RunStatus(path, {"split": "train"}, ["a", "b"])
    assert json.loads(path.read_text())["counts"] == {"queued": 2, "running": 0, "done": 0, "error": 0}

    status.start("a")
    policy = TrackingPolicy(FakePolicy(100), status, "a")
    for prompt_len in (1000, 1300):  # turn 2 re-reads turn 1's 1000 + 100 tokens, plus 200 new
        asyncio.run(policy(tinker.ModelInput.from_ints([0] * prompt_len), [0]))
    status.finish("a", {"error": None, "stop_reason": "tool_stopped", "submitted": True, "submit_turn": 2, "tests_passed": True})

    snapshot = json.loads(path.read_text())
    task = next(t for t in snapshot["tasks"] if t["instance_id"] == "a")
    assert (task["turns"], task["input_new"], task["input_cached"], task["output"]) == (2, 1200, 1100, 200)
    assert snapshot["counts"] == {"queued": 1, "running": 0, "done": 1, "error": 0}
    assert snapshot["totals"]["submitted"] == snapshot["totals"]["tests_passed"] == 1
    assert "_prev_total" not in task

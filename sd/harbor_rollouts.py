"""SWE-bench Verified rollouts (Harbor tasks in Modal sandboxes) for any token-level policy.

    uv run python -m sd.harbor_rollouts --policy sd.tinker_policy:make_tinker_policy \
        --n-tasks 10 --output runs/baseline/trajectories.jsonl

The policy is loaded by import path and must be a factory taking (model_name, max_tokens,
temperature) and returning a tinker_cookbook TokenCompleter, so this module doesn't depend on
any particular backend.

Every trajectory is graded after the episode ends, whatever ended it, by running the task's
tests/test.sh on the sandbox's final state. That gives "tests pass" independent of whether
the agent submitted (the talk's test pass rate), alongside whether and when it submitted.
"""

import asyncio
import importlib
import json
import logging
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import modal
import typer
from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask, load_harbor_tasks
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborReward
from tinker_cookbook.renderers.base import message_to_jsonable
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import STOP_METRIC_PREFIX, Trajectory
from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox
from tinker_cookbook.tool_use import ToolResult, build_agent_tool_env, simple_tool_result, tool

from sd.prompts import INSTANCE_TEMPLATE, SYSTEM_PROMPT, observation
from sd.renderer import Qwen35AgentRenderer, get_agent_renderer

logger = logging.getLogger(__name__)

DATASET = "swebench-verified-1.0/swebench-verified"  # under ~/.cache/harbor/tasks/
# Commands run in a non-interactive shell, which doesn't reach the image's `conda activate testbed`.
CONDA_PREFIX = "source /opt/miniconda3/bin/activate && conda activate testbed && "


async def create_sandbox(task: HarborTask, timeout: int = 3600) -> ModalSandbox:
    """Modal sandbox from the task's Dockerfile, with the CPU/memory its task.toml asks for.

    The cookbook's default factory requests nothing, which leaves Modal's small default
    allocation; SWE-bench test runs and test.sh's reinstall need more.
    """
    env_dir = task.task_dir / "environment"
    resources = task.config.get("environment", {})
    image = modal.Image.from_dockerfile(path=str(env_dir / "Dockerfile"), context_dir=str(env_dir))
    app = await modal.App.lookup.aio("sd-swebench-rollouts", create_if_missing=True)
    sandbox = await modal.Sandbox.create.aio(
        app=app,
        image=image,
        timeout=timeout,
        cpu=float(resources.get("cpus", 1)),
        memory=parse_memory_mb(resources.get("memory", "4G")),
    )
    return ModalSandbox(timeout=timeout, image=image, app=app, sandbox=sandbox)


def parse_memory_mb(value: str) -> int:
    """'4G' -> 4096, '512M' -> 512."""
    value = value.strip().upper()
    if value.endswith("G"):
        return int(float(value[:-1]) * 1024)
    if value.endswith("M"):
        return int(float(value[:-1]))
    return int(value)


class SweTools:
    """bash (mini-swe-agent style observations) + task_complete (ends the episode)."""

    def __init__(self, sandbox: SandboxInterface, command_timeout: int = 60):
        self.sandbox = sandbox
        self.command_timeout = command_timeout

    @tool
    async def bash(self, command: Annotated[str, "The bash command to execute"]) -> ToolResult:
        """Execute a bash command"""
        result = await self.sandbox.run_command(
            CONDA_PREFIX + command, workdir="/testbed", timeout=self.command_timeout
        )
        return simple_tool_result(observation(result.exit_code, result.stdout + result.stderr))

    @tool
    async def task_complete(self) -> ToolResult:
        """Submit your work. Call this once you have finished the task; you cannot continue afterwards."""
        return simple_tool_result("Task submitted.", should_stop=True)

    def all(self):
        return [self.bash, self.task_complete]


def initial_messages(task: HarborTask, renderer: Qwen35AgentRenderer, tools: SweTools):
    # Qwen's template serializes the full OpenAI tool object ({"type": "function", ...}).
    specs = [{"type": "function", "function": t.to_spec()} for t in tools.all()]
    prefix = renderer.create_conversation_prefix_with_tools(specs, system_prompt=SYSTEM_PROMPT)
    return prefix + [{"role": "user", "content": INSTANCE_TEMPLATE.format(task=task.instruction)}]


async def no_grade(history) -> tuple[float, dict[str, float]]:
    """In-episode grading is skipped; every trajectory is graded after the rollout instead."""
    return 0.0, {}


def submit_turn(messages: list[dict]) -> int | None:
    """1-based index of the assistant turn that called task_complete."""
    turn = 0
    for m in messages:
        if m["role"] == "assistant":
            turn += 1
            if any(tc.function.name == "task_complete" for tc in m.get("tool_calls", [])):
                return turn
    return None


def to_record(task, config, trajectory: Trajectory, messages, grade, elapsed, error=None) -> dict:
    transitions = trajectory.transitions if trajectory else []
    stop_reasons = [
        k.removeprefix(STOP_METRIC_PREFIX) for k in (transitions[-1].metrics if transitions else {})
        if k.startswith(STOP_METRIC_PREFIX)
    ]
    turn = submit_turn(messages)
    return {
        "instance_id": task.task_name,
        **config,
        "stop_reason": stop_reasons[0] if stop_reasons else None,
        "error": error,
        "submitted": turn is not None,
        "submit_turn": turn,
        "n_turns": len(transitions),
        "tests_passed": grade.get("test_passed") == 1.0 if grade else None,
        "grade": grade,
        "elapsed_s": round(elapsed, 1),
        "messages": [message_to_jsonable(m) for m in messages],
        # Per turn: prompt length and the exact sampled tokens/logprobs. Prompt tokens are not
        # stored (quadratic); re-render `messages[:k]` with the same renderer to recover them.
        "turns": [
            {
                "prompt_len": t.ob.length,
                "tokens": list(t.ac.tokens),
                "logprobs": list(t.ac.maybe_logprobs or []),
                "stop_reason": t.ac.stop_reason,
            }
            for t in transitions
        ],
    }


async def run_task(task, policy, renderer, config, max_turns, max_context, max_tokens) -> dict:
    start = time.monotonic()
    sandbox = trajectory = grade = error = None
    messages: list = []
    try:
        sandbox = await create_sandbox(task)
        tools = SweTools(sandbox)
        env = build_agent_tool_env(
            renderer=renderer,
            tools=tools.all(),
            initial_messages=initial_messages(task, renderer, tools),
            reward_fn=no_grade,
            max_turns=max_turns,
            max_trajectory_tokens=max_context,
            max_generation_tokens=max_tokens,
            # Like mini-swe-agent: tell the model about a malformed tool call, up to 3 in a row.
            parse_error_policy=ParseErrorPolicy(max_consecutive=3),
        )
        trajectory = await do_single_rollout(policy, env)
        messages = env.message_env.history
        _, grade = await HarborReward(tests_dir=task.task_dir / "tests", sandbox=sandbox, grader_timeout=1800)(messages)
    except Exception as e:
        logger.exception("Task %s failed", task.task_name)
        error = f"{type(e).__name__}: {e}"
    finally:
        if sandbox is not None:
            try:
                await sandbox.cleanup()
            except Exception as e:
                logger.warning("Sandbox cleanup failed for %s: %s", task.task_name, e)
    record = to_record(task, config, trajectory, messages, grade, time.monotonic() - start, error)
    logger.info(
        "%s: stop=%s submitted=%s turn=%s tests_passed=%s",
        task.task_name, record["stop_reason"], record["submitted"], record["submit_turn"], record["tests_passed"],
    )
    return record


def load_policy_factory(spec: str) -> Callable[..., TokenCompleter]:
    """'package.module:function' -> the factory; it must take (model_name, max_tokens, temperature)."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise typer.BadParameter(f"--policy must look like 'package.module:function', got {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


async def main_async(policy_spec, model_name, n_tasks, seed, output, max_turns, max_context, max_tokens, temperature):
    tasks = load_harbor_tasks(DATASET)
    random.Random(seed).shuffle(tasks)
    tasks = tasks[:n_tasks]

    renderer = get_agent_renderer(model_name)
    policy = load_policy_factory(policy_spec)(model_name=model_name, max_tokens=max_tokens, temperature=temperature)
    config = {
        "policy": policy_spec,
        "model_name": model_name,
        "renderer": type(renderer).__name__,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_context": max_context,
        "max_turns": max_turns,
        "seed": seed,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as f:
        for coro in asyncio.as_completed(
            [run_task(t, policy, renderer, config, max_turns, max_context, max_tokens) for t in tasks]
        ):
            f.write(json.dumps(await coro) + "\n")
            f.flush()
    print(f"Wrote {len(tasks)} trajectories to {output}")


def main(
    policy: str = typer.Option(
        ..., help="Policy factory as 'package.module:function', e.g. sd.tinker_policy:make_tinker_policy"
    ),
    model_name: str = "Qwen/Qwen3.5-9B",
    n_tasks: int = 10,
    seed: int = 0,
    output: Path = Path("runs/baseline/trajectories.jsonl"),
    max_turns: int = 150,
    max_context: int = 65536,
    max_tokens: int = 16384,
    temperature: float = 1.0,
) -> None:
    """Temperature 1.0 (no top-p/top-k) keeps sampled logprobs equal to the model's own distribution."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(main_async(policy, model_name, n_tasks, seed, output, max_turns, max_context, max_tokens, temperature))


if __name__ == "__main__":
    typer.run(main)

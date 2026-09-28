"""Offline tests: the cookbook's real rollout loop with a fake sampler and a fake sandbox."""

import asyncio
import json
from pathlib import Path

import pytest
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.sandbox.sandbox_interface import SandboxResult
from tinker_cookbook.tool_use import build_agent_tool_env

from sd import harbor_rollouts
from sd.harbor_rollouts import SweTools, initial_messages, no_grade, to_record
from sd.renderer import get_agent_renderer

MODEL = "Qwen/Qwen3.5-9B"


def call(thought: str, text: str, name: str, command: str | None = None) -> str:
    params = f"<parameter=command>\n{command}\n</parameter>\n" if command is not None else ""
    return f"{thought}\n</think>\n\n{text}\n\n<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


LS = call("Let me look at the repo first.", "Listing files.", "bash", "ls -la")
SUBMIT = call("The fix is done.", "Submitting.", "task_complete")


class FakeSandbox:
    def __init__(self):
        self.commands = []

    async def run_command(self, command, workdir=None, timeout=60, max_output_bytes=None):
        self.commands.append((command, workdir))
        return SandboxResult(stdout="a\nb\n", stderr="", exit_code=0)


class FakePolicy:
    """Plays back scripted completions and records every prompt it was given."""

    def __init__(self, tokenizer, texts):
        self.tokenizer, self.texts, self.prompts = tokenizer, list(texts), []

    async def __call__(self, model_input, stop, *, max_tokens=None):
        self.prompts.append(model_input.to_ints())
        tokens = self.tokenizer.encode(self.texts.pop(0), add_special_tokens=False)
        tokens.append(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        return TokensWithLogprobs(tokens=tokens, maybe_logprobs=[-0.1] * len(tokens))


@pytest.fixture(scope="module")
def renderer():
    return get_agent_renderer(MODEL)


def rollout(renderer, texts):
    task = HarborTask(task_name="demo__demo-1", instruction="Fix the bug.", task_dir=Path("/nonexistent"))
    sandbox = FakeSandbox()
    tools = SweTools(sandbox)
    env = build_agent_tool_env(
        renderer=renderer, tools=tools.all(), initial_messages=initial_messages(task, renderer, tools),
        reward_fn=no_grade, max_turns=150,
    )
    policy = FakePolicy(renderer.tokenizer, texts)
    trajectory = asyncio.run(do_single_rollout(policy, env))
    return task, sandbox, policy, trajectory, env.message_env.history


def test_task_complete_ends_episode_and_is_recorded(renderer):
    task, sandbox, _, trajectory, messages = rollout(renderer, [LS, SUBMIT])
    assert sandbox.commands == [(harbor_rollouts.CONDA_PREFIX + "ls -la", "/testbed")]
    record = to_record(task, {"model_name": MODEL}, trajectory, messages, {"test_passed": 1.0}, 1.0)
    assert record["submitted"] and record["submit_turn"] == 2 and record["n_turns"] == 2
    assert record["stop_reason"] == "tool_stopped"
    assert record["tests_passed"] is True
    assert all(len(t["tokens"]) == len(t["logprobs"]) > 0 for t in record["turns"])
    json.dumps(record)  # the JSONL line must serialize


def test_next_prompt_extends_previous_prompt_and_completion(renderer):
    """Token-exact history: turn-2 prompt starts with turn-1 prompt + turn-1 sampled tokens."""
    _, _, policy, trajectory, _ = rollout(renderer, [LS, SUBMIT])
    p1, p2 = policy.prompts
    prefix = p1 + list(trajectory.transitions[0].ac.tokens)
    assert p2[: len(prefix)] == prefix


def test_prompt_matches_official_chat_template(renderer):
    """The turn-2 prompt is exactly what Qwen's HF chat template produces."""
    _, _, policy, _, messages = rollout(renderer, [LS, SUBMIT])
    tools = [{"type": "function", "function": t.to_spec()} for t in SweTools(None).all()]
    hf_messages = [
        {"role": "system", "content": harbor_rollouts.SYSTEM_PROMPT},
        messages[1],
        {
            "role": "assistant",
            "reasoning_content": "Let me look at the repo first.",
            "content": "Listing files.",
            "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": {"command": "ls -la"}}}],
        },
        {"role": "tool", "content": messages[3]["content"]},
    ]
    expected = renderer.tokenizer.apply_chat_template(hf_messages, tools=tools, tokenize=False, add_generation_prompt=True)
    assert renderer.tokenizer.decode(policy.prompts[1]) == expected

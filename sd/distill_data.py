"""Offline self-distillation examples: hinted teacher vs unhinted student on stored trajectories.

    uv run python -m sd.distill_data runs/train250/trajectories.jsonl --output runs/distill/examples.jsonl

For each trajectory that didn't submit by the turn limit, pick one assistant turn t in
[hint_min, hint_max] (seeded per task). Then:

- student sequence: the conversation up to turn t, rendered exactly as the model saw it, followed by
  the tokens the model actually sampled at turn t (the target)
- teacher sequence: the same conversation plus a hint user message right before turn t, followed by
  the same target tokens

Training (next step) matches the student's distribution to the teacher's on the target tokens only.
Both prompts end with the same generation suffix, so the target sits at the end of both sequences.
"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import typer
from tinker_cookbook.renderers import Message, ToolCall

from sd.renderer import Qwen35AgentRenderer, get_agent_renderer

TURN_LIMIT = 40
HINT = (
    "Note: you're close to your {limit}-turn limit, with about {left} {turns} left. You often keep "
    "exploring and forget to submit. Wrap up and call `task_complete`."
)


class HintChangesContext(ValueError):
    """Adding the hint message re-renders earlier context, so the teacher differs by more than the hint.

    Qwen3.5's template gives an assistant turn without thinking an empty <think></think> block only
    after the last user message; the hint becomes the last user message, so earlier no-thinking
    turns lose it in the teacher's rendering. Such examples are skipped to keep the hint the only
    difference between teacher and student.
    """


@dataclass
class Example:
    instance_id: str
    hint_turn: int  # 1-based assistant turn the hint comes right before
    hint: str
    student_prompt: list[int]
    teacher_prompt: list[int]
    target: list[int]  # the tokens the model sampled at hint_turn
    target_logprobs: list[float]  # their logprobs under the policy that sampled them


def hint_text(turn: int, limit: int = TURN_LIMIT) -> str:
    left = limit - turn + 1  # turns remaining, counting the one about to be taken
    return HINT.format(limit=limit, left=left, turns="turn" if left == 1 else "turns")


def from_jsonable(message: dict) -> Message:
    """A message as stored in the JSONL back into a renderer message."""
    message = dict(message)
    if message.get("tool_calls"):
        message["tool_calls"] = [ToolCall.model_validate(tc) for tc in message["tool_calls"]]
    return message


def is_late(record: dict, limit: int = TURN_LIMIT) -> bool:
    """Didn't call task_complete within the turn limit (including never)."""
    return not (record["submitted"] and record["submit_turn"] <= limit)


def pick_hint_turn(record: dict, seed: int, hint_min: int, hint_max: int) -> int | None:
    """Uniform in [hint_min, min(hint_max, n_turns)], seeded per task so it doesn't depend on order."""
    hi = min(hint_max, record["n_turns"])
    if hi < hint_min:
        return None
    return random.Random(f"{seed}:{record['instance_id']}").randint(hint_min, hi)


def build_example(record: dict, turn: int, renderer: Qwen35AgentRenderer) -> Example:
    messages = [from_jsonable(m) for m in record["messages"]]
    assistant_idx = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    prefix = messages[: assistant_idx[turn - 1]]
    hint = hint_text(turn)

    student_prompt = renderer.build_generation_prompt(prefix).to_ints()
    teacher_prompt = renderer.build_generation_prompt(prefix + [{"role": "user", "content": hint}]).to_ints()
    stored = record["turns"][turn - 1]

    # The re-rendered prompt must be exactly what the model saw when it sampled this turn.
    if len(student_prompt) != stored["prompt_len"]:
        raise ValueError(f"{record['instance_id']} turn {turn}: rendered {len(student_prompt)} tokens, "
                         f"the model saw {stored['prompt_len']}")
    # Teacher = student without its generation suffix, + the hint message, + the same suffix.
    suffix = renderer.build_generation_prompt([]).to_ints()  # e.g. "<|im_start|>assistant\n<think>\n"
    body = student_prompt[: len(student_prompt) - len(suffix)]
    if teacher_prompt[: len(body)] != body or teacher_prompt[-len(suffix):] != suffix:
        raise HintChangesContext(f"{record['instance_id']} turn {turn}: teacher prompt doesn't extend the student's")

    return Example(
        instance_id=record["instance_id"],
        hint_turn=turn,
        hint=hint,
        student_prompt=student_prompt,
        teacher_prompt=teacher_prompt,
        target=list(stored["tokens"]),
        target_logprobs=list(stored["logprobs"]),
    )


def build_examples(records: list[dict], renderer: Qwen35AgentRenderer, seed: int = 0,
                   hint_min: int = 30, hint_max: int = 39) -> tuple[list[Example], list[str]]:
    """Examples, plus the tasks skipped because the hint changed more than the context's end."""
    examples, skipped = [], []
    for record in records:
        if record["error"] or not is_late(record):
            continue
        turn = pick_hint_turn(record, seed, hint_min, hint_max)
        if turn is None:
            continue
        try:
            examples.append(build_example(record, turn, renderer))
        except HintChangesContext:
            skipped.append(record["instance_id"])
    return examples, skipped


def load_records(path: Path) -> list[dict]:
    """Latest record per task (a resumed run can append retried tasks)."""
    latest = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        latest[record["instance_id"]] = record
    return list(latest.values())


def main(trajectories: Path, output: Path = Path("runs/distill/examples.jsonl"), seed: int = 0,
         hint_min: int = 30, hint_max: int = 39) -> None:
    records = load_records(trajectories)
    renderer = get_agent_renderer(records[0]["model_name"])
    examples, skipped = build_examples(records, renderer, seed, hint_min, hint_max)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(asdict(e)) + "\n" for e in examples))
    print(f"{len(examples)} examples from {len(records)} trajectories -> {output}")
    if skipped:
        print(f"skipped {len(skipped)} where the hint changed earlier context: {', '.join(skipped)}")


if __name__ == "__main__":
    typer.run(main)

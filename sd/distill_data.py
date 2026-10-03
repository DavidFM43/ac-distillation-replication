"""Offline self-distillation examples: hinted teacher vs unhinted student on stored trajectories.

    uv run python -m sd.distill_data runs/train250/trajectories.jsonl --output runs/distill/examples_all.jsonl
    uv run python -m sd.distill_data runs/train250/trajectories.jsonl --n-turns 1 \
        --output runs/distill/examples.jsonl

For each trajectory that didn't submit by the turn limit, pick one assistant turn t in
[hint_min, hint_max] (seeded per task). Then:

- student sequence: the conversation up to turn t, rendered exactly as the model saw it, followed by
  the target: turns t..t+n_turns-1 as the model actually sampled them, with the tool outputs between
  them (fewer turns if the trajectory ends sooner; turns past the limit count too). Without n_turns,
  the target is the whole rest of the trajectory. Supervision also stops early at a turn after which
  re-rendering the history changes it (a turn without thinking), since one sequence can't match both
  what the model saw before and after.
- teacher sequence: the same conversation plus a hint user message right before turn t, followed by
  the same target tokens

`target_mask` marks the assistant turns' tokens (1) versus tool outputs and chat formatting between
turns (0). Training (next step) matches the student's distribution to the teacher's on the masked-in
tokens only. Both prompts end with the same generation suffix, so the target sits at the end of both
sequences.

With several turns, the target is the last supervised turn's prompt as the model saw it, plus that
turn's sampled tokens. Each prompt re-renders earlier turns from their text, and sometimes (about 4%
of turns) the model had sampled a different token split of the same text than re-encoding gives, so
earlier supervised turns use the tokens the model actually had in its context.
"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import typer
from tinker_cookbook.renderers import Message, ToolCall

from sd.renderer import Qwen35AgentRenderer, get_agent_renderer

TURN_LIMIT = 40
IM_END = "<|im_end|>"
HINT = (
    "Note: you're close to your {limit}-turn limit, with about {left} {turns} left. You often keep "
    "exploring and forget to submit. Wrap up and call `task_complete`."
)


class SkipExample(ValueError):
    """The trajectory can't give a clean example; it's skipped and reported."""


class HintChangesContext(SkipExample):
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
    n_supervised: int  # assistant turns in the target: hint_turn..hint_turn+n_supervised-1
    hint: str
    student_prompt: list[int]
    teacher_prompt: list[int]
    target: list[int]  # the supervised turns' sampled tokens, with the tool outputs between them
    target_mask: list[int]  # per target token: 1 for the assistant turns, 0 for tool outputs/formatting


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


def rendered_prompt(record: dict, messages: list[Message], assistant_idx: list[int], turn: int,
                    renderer: Qwen35AgentRenderer) -> list[int]:
    """The prompt for assistant turn `turn`, checked to be exactly what the model saw when it sampled it."""
    prompt = renderer.build_generation_prompt(messages[: assistant_idx[turn - 1]]).to_ints()
    if len(prompt) != record["turns"][turn - 1]["prompt_len"]:
        raise ValueError(f"{record['instance_id']} turn {turn}: rendered {len(prompt)} tokens, "
                         f"the model saw {record['turns'][turn - 1]['prompt_len']}")
    return prompt


def build_example(record: dict, turn: int, renderer: Qwen35AgentRenderer, n_turns: int | None = None) -> Example:
    messages = [from_jsonable(m) for m in record["messages"]]
    assistant_idx = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    last = min(len(assistant_idx), len(record["turns"]))
    if n_turns is not None:
        last = min(last, turn + n_turns - 1)
    hint = hint_text(turn)
    hint_message = {"role": "user", "content": hint}
    im_end = renderer.tokenizer.convert_tokens_to_ids(IM_END)

    student_prompt = rendered_prompt(record, messages, assistant_idx, turn, renderer)
    teacher_prompt = renderer.build_generation_prompt(messages[: assistant_idx[turn - 1]] + [hint_message]).to_ints()
    # Teacher = student without its generation suffix, + the hint message, + the same suffix.
    suffix = renderer.build_generation_prompt([]).to_ints()  # e.g. "<|im_start|>assistant\n<think>\n"
    body = student_prompt[: len(student_prompt) - len(suffix)]
    if teacher_prompt[: len(body)] != body or teacher_prompt[-len(suffix):] != suffix:
        raise HintChangesContext(f"{record['instance_id']} turn {turn}: teacher prompt doesn't extend the student's")

    # The target: the last supervised turn's prompt past the student prompt, + that turn's sampled tokens.
    # Each turn's prompt must end with the generation suffix at its stored length. Where it doesn't, the
    # later history re-renders that turn (a turn without thinking gets an empty think block once it's
    # history), so supervision stops at it: its own prompt is still exactly what the model saw.
    while True:
        last_prompt = rendered_prompt(record, messages, assistant_idx, last, renderer) if last > turn else student_prompt
        starts = [record["turns"][t - 1]["prompt_len"] for t in range(turn, last + 1)]
        ok = [last_prompt[n - len(suffix): n] == suffix for n in starts]
        if all(ok):
            break
        if ok.index(False) == len(ok) - 1:
            raise SkipExample(f"{record['instance_id']} turn {last}: prompt doesn't end with the generation suffix")
        last = turn + ok.index(False)
    if last_prompt[: len(student_prompt)] != student_prompt:
        raise SkipExample(f"{record['instance_id']} turn {last}: prompt doesn't extend turn {turn}'s")
    last_tokens = record["turns"][last - 1]["tokens"]
    target = last_prompt[len(student_prompt):] + last_tokens

    # Each supervised turn runs from where its prompt ended through its end token; what's between (tool
    # outputs, formatting) is masked out.
    mask = [0] * len(target)
    for n in starts:
        i = n - len(student_prompt)
        end = target.index(im_end, i) + 1 if im_end in target[i:] else len(target)
        mask[i:end] = [1] * (end - i)

    # The teacher's own rendering of the later turns must match too: the hint changes nothing after it.
    teacher_last = renderer.build_generation_prompt(
        messages[: assistant_idx[turn - 1]] + [hint_message] + messages[assistant_idx[turn - 1]: assistant_idx[last - 1]]
    ).to_ints()
    if teacher_last != teacher_prompt + target[: len(target) - len(last_tokens)]:
        raise HintChangesContext(f"{record['instance_id']} turn {turn}: the hint re-renders turns after it")

    return Example(
        instance_id=record["instance_id"],
        hint_turn=turn,
        n_supervised=last - turn + 1,
        hint=hint,
        student_prompt=student_prompt,
        teacher_prompt=teacher_prompt,
        target=target,
        target_mask=mask,
    )


def build_examples(records: list[dict], renderer: Qwen35AgentRenderer, seed: int = 0, hint_min: int = 30,
                   hint_max: int = 39, n_turns: int | None = None) -> tuple[list[Example], dict[str, str]]:
    """Examples, plus the skipped tasks with why (e.g. the hint changed more than the context's end)."""
    examples, skipped = [], {}
    for record in records:
        if record["error"] or not is_late(record):
            continue
        turn = pick_hint_turn(record, seed, hint_min, hint_max)
        if turn is None:
            continue
        try:
            examples.append(build_example(record, turn, renderer, n_turns))
        except SkipExample as e:
            skipped[record["instance_id"]] = str(e)
    return examples, skipped


def load_records(path: Path) -> list[dict]:
    """Latest record per task (a resumed run can append retried tasks)."""
    latest = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        latest[record["instance_id"]] = record
    return list(latest.values())


def main(trajectories: Path, output: Path = Path("runs/distill/examples.jsonl"), seed: int = 0,
         hint_min: int = 30, hint_max: int = 39,
         n_turns: int | None = typer.Option(
             None, help="Assistant turns to supervise, starting at the hint turn (default: the rest of the trajectory)"
         )) -> None:
    records = load_records(trajectories)
    renderer = get_agent_renderer(records[0]["model_name"])
    examples, skipped = build_examples(records, renderer, seed, hint_min, hint_max, n_turns)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(asdict(e)) + "\n" for e in examples))
    print(f"{len(examples)} examples from {len(records)} trajectories -> {output}")
    for reason in skipped.values():
        print(f"skipped {reason}")


if __name__ == "__main__":
    typer.run(main)

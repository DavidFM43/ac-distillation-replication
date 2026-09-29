"""Offline tests: building hinted-teacher / student examples from a stored trajectory."""

import pytest
from tinker_cookbook.renderers.base import message_to_jsonable

from sd.distill_data import (HintChangesContext, build_example, build_examples, hint_text, is_late,
                             pick_hint_turn)
from sd.renderer import get_agent_renderer

MODEL = "Qwen/Qwen3.5-9B"
IM_END = "<|im_end|>"


@pytest.fixture(scope="module")
def renderer():
    return get_agent_renderer(MODEL)


def turn_text(i: int, thinking: bool = True) -> str:
    call = f"<tool_call>\n<function=bash>\n<parameter=command>\necho {i}\n</parameter>\n</function>\n</tool_call>"
    return f"Step {i} reasoning.\n</think>\n\nRunning step {i}.\n\n{call}" if thinking else f"Running step {i}.\n\n{call}"


def make_record(renderer, n_turns: int, no_thinking_turns=(), **overrides) -> dict:
    """A stored trajectory as sd.harbor_rollouts writes it: messages plus per-turn tokens."""
    messages = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Fix the bug."}]
    turns = []
    for i in range(1, n_turns + 1):
        prompt = renderer.build_generation_prompt(messages).to_ints()
        tokens = renderer.tokenizer.encode(turn_text(i, i not in no_thinking_turns), add_special_tokens=False)
        tokens.append(renderer.tokenizer.convert_tokens_to_ids(IM_END))
        assistant, _ = renderer.parse_response(tokens)
        call_id = assistant["tool_calls"][0].id or "call_0"
        messages += [assistant, {"role": "tool", "content": f"out {i}", "tool_call_id": call_id, "name": "bash"}]
        turns.append({"prompt_len": len(prompt), "tokens": tokens, "logprobs": [-0.5] * len(tokens)})
    record = {"instance_id": "repo__repo-1", "model_name": MODEL, "error": None, "n_turns": n_turns,
              "submitted": False, "submit_turn": None, "messages": [message_to_jsonable(m) for m in messages],
              "turns": turns}
    return record | overrides


def test_teacher_is_student_plus_hint_and_targets_line_up(renderer):
    record = make_record(renderer, 6)
    ex = build_example(record, 4, renderer)
    suffix = renderer.build_generation_prompt([]).to_ints()
    body = ex.student_prompt[: len(ex.student_prompt) - len(suffix)]
    hint_block = ex.teacher_prompt[len(body): len(ex.teacher_prompt) - len(suffix)]

    assert len(ex.student_prompt) == record["turns"][3]["prompt_len"]  # exactly what the model saw
    assert ex.teacher_prompt == body + hint_block + suffix
    assert "40-turn limit" in renderer.tokenizer.decode(hint_block) and "37 turns left" in ex.hint
    assert ex.target == record["turns"][3]["tokens"] and len(ex.target_logprobs) == len(ex.target)


def test_hint_turn_is_seeded_per_task_and_capped_by_length(renderer):
    record = make_record(renderer, 5)
    turns = {pick_hint_turn(record, seed, 2, 9) for seed in range(50)}
    assert turns <= {2, 3, 4, 5} and len(turns) > 1  # capped at n_turns, and actually varies with the seed
    assert pick_hint_turn(record, 7, 2, 9) == pick_hint_turn(record, 7, 2, 9)
    assert pick_hint_turn(record, 0, 6, 9) is None  # trajectory too short for the range


def test_only_late_trajectories_become_examples(renderer):
    late = make_record(renderer, 4)
    early = make_record(renderer, 4, instance_id="repo__repo-2", submitted=True, submit_turn=3)
    errored = make_record(renderer, 4, instance_id="repo__repo-3", error="SandboxError")
    examples, skipped = build_examples([late, early, errored], renderer, hint_min=2, hint_max=3)
    assert [e.instance_id for e in examples] == ["repo__repo-1"] and skipped == []
    assert is_late(make_record(renderer, 1, submitted=True, submit_turn=41))


def test_hint_that_rerenders_earlier_context_is_skipped(renderer):
    # A turn without thinking gets an empty <think></think> only after the last user message, so
    # the hint (a new last user message) re-renders it: the teacher would differ beyond the hint.
    record = make_record(renderer, 4, no_thinking_turns=(2,))
    with pytest.raises(HintChangesContext):
        build_example(record, 4, renderer)
    examples, skipped = build_examples([record], renderer, hint_min=4, hint_max=4)
    assert examples == [] and skipped == ["repo__repo-1"]


def test_hint_counts_the_current_turn():
    assert "about 4 turns left" in hint_text(37) and "about 1 turn left" in hint_text(40)

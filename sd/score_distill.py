"""Score distillation examples with the hinted teacher and the unhinted student, in PyTorch.

    uv run python -m sd.score_distill runs/distill/all_after_hint.jsonl \
        --output runs/distill/all_after_hint_scored.jsonl

Both are the same model (bf16, transformers), so the hint is the only difference between them:
no second engine's numerics end up in the comparison. For each example, at every target position:

- `teacher_topk_ids`, `teacher_topk_logprobs`: the teacher's top-K tokens and their full-vocabulary
  log-probs, given `teacher_prompt` + the target so far (best first)
- `student_topk_logprobs`: the student's full-vocabulary log-probs of those same K tokens, given
  `student_prompt` + the target so far
- `teacher_target_logprob`, `student_target_logprob`: each one's log-prob of the token actually written

Each list has one row per target token, aligned with `target` and `target_mask`. The top-K KL that
training minimizes follows from these (teacher renormalized over its K, student as is).

The output is the input with these fields added. It's appended to one example at a time, and a
rerun skips the examples already in it, so an interrupted run resumes.
"""

import json
from pathlib import Path

import torch
import typer
from transformers import AutoModelForImageTextToText

MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"  # the commit deploy/qwen35_9b.yaml serves


def load_model(model_name: str = MODEL, revision: str | None = REVISION, device: str = "cuda"):
    model = AutoModelForImageTextToText.from_pretrained(model_name, revision=revision, dtype=torch.bfloat16)
    return model.to(device).eval()


@torch.inference_mode()
def target_logprobs(model, prompt: list[int], target: list[int], chunk: int = 1024):
    """Yields (row offset, full-vocabulary log-probs) for the target's rows, a chunk of rows at a time.

    Row i is the distribution of target[i] given the prompt and target[:i]. Only these rows go
    through the LM head, a chunk at a time, so the ~248k-wide logits never exist for the whole sequence.
    """
    seq = prompt + target
    device = model.lm_head.weight.device
    # The model reads up to the second-to-last token; its last T hidden states predict the target.
    hidden = model.model.language_model(input_ids=torch.tensor([seq[:-1]], device=device)).last_hidden_state[0]
    hidden = hidden[-len(target):]
    for start in range(0, len(target), chunk):
        logits = model.lm_head(hidden[start:start + chunk]).float()
        yield start, torch.log_softmax(logits, dim=-1)


def score_example(model, example: dict, k: int = 20, chunk: int = 1024) -> dict:
    """The teacher's top-K and the student's log-probs of them at every target position."""
    target = example["target"]
    target_ids = torch.tensor(target, device=model.lm_head.weight.device).unsqueeze(1)

    topk_ids, topk_lp, teacher_target_lp = [], [], []
    for start, lp in target_logprobs(model, example["teacher_prompt"], target, chunk):
        values, ids = lp.topk(k, dim=-1)
        topk_ids.append(ids)
        topk_lp.append(values)
        teacher_target_lp.append(lp.gather(1, target_ids[start:start + len(lp)]).squeeze(1))
    topk_ids = torch.cat(topk_ids)

    student_lp, student_target_lp = [], []
    for start, lp in target_logprobs(model, example["student_prompt"], target, chunk):
        student_lp.append(lp.gather(1, topk_ids[start:start + len(lp)]))
        student_target_lp.append(lp.gather(1, target_ids[start:start + len(lp)]).squeeze(1))

    return {
        "teacher_topk_ids": topk_ids.tolist(),
        "teacher_topk_logprobs": torch.cat(topk_lp).tolist(),
        "student_topk_logprobs": torch.cat(student_lp).tolist(),
        "teacher_target_logprob": torch.cat(teacher_target_lp).tolist(),
        "student_target_logprob": torch.cat(student_target_lp).tolist(),
    }


def scored_ids(path: Path) -> set[str]:
    """instance_ids already in the output; a line cut off by an interrupted write is dropped."""
    if not path.exists():
        return set()
    ids, lines = set(), path.read_text().splitlines(keepends=True)
    complete = [line for line in lines if line.endswith("\n")]
    if len(complete) < len(lines):
        path.write_text("".join(complete))
    for line in complete:
        ids.add(json.loads(line)["instance_id"])
    return ids


def main(examples: Path, output: Path = typer.Option(..., help="Scored copy of the examples (JSONL)"),
         model_name: str = MODEL, revision: str = REVISION, k: int = 20, chunk: int = 1024,
         device: str = "cuda") -> None:
    done = scored_ids(output)
    todo = [e for e in map(json.loads, examples.read_text().splitlines()) if e["instance_id"] not in done]
    print(f"{len(done)} already scored, {len(todo)} to go -> {output}")
    model = load_model(model_name, revision, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    for i, example in enumerate(todo, 1):
        example |= score_example(model, example, k, chunk)
        with output.open("a") as f:
            f.write(json.dumps(example) + "\n")
        print(f"[{i}/{len(todo)}] {example['instance_id']}: {len(example['target'])} target tokens", flush=True)


if __name__ == "__main__":
    typer.run(main)

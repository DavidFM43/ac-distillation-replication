"""Offline tests: teacher/student scoring, on a tiny random Qwen3.5 on CPU."""

import json

import pytest
import torch
from transformers import AutoModelForImageTextToText, Qwen3_5Config

from sd.score_distill import main, score_example

K = 5


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = Qwen3_5Config(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2,
                         linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32),
    )
    return AutoModelForImageTextToText.from_config(config).float().eval()


def make_example(instance_id="t-1", n_prompt=12, n_hint=3, n_target=9) -> dict:
    g = torch.Generator().manual_seed(1)
    student_prompt = torch.randint(0, 64, (n_prompt,), generator=g).tolist()
    hint = torch.randint(0, 64, (n_hint,), generator=g).tolist()
    target = torch.randint(0, 64, (n_target,), generator=g).tolist()
    return {"instance_id": instance_id, "student_prompt": student_prompt,
            "teacher_prompt": student_prompt[:-2] + hint + student_prompt[-2:], "target": target,
            "target_mask": [1] * n_target}


def full_logprobs(model, prompt, target):
    """Reference: log-probs of the target rows from the full model's logits over the whole sequence."""
    seq = prompt + target
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([seq])).logits[0]
    return torch.log_softmax(logits.float(), -1)[len(prompt) - 1:len(seq) - 1]


def test_scores_match_the_full_forward_pass(model):
    example = make_example()
    scores = score_example(model, example, k=K, chunk=4)  # chunks smaller than the target
    teacher = full_logprobs(model, example["teacher_prompt"], example["target"])
    student = full_logprobs(model, example["student_prompt"], example["target"])
    target = torch.tensor(example["target"]).unsqueeze(1)

    ids = torch.tensor(scores["teacher_topk_ids"])
    assert ids.shape == (len(example["target"]), K)
    assert torch.equal(ids, teacher.topk(K, -1).indices)
    torch.testing.assert_close(torch.tensor(scores["teacher_topk_logprobs"]), teacher.gather(1, ids))
    torch.testing.assert_close(torch.tensor(scores["student_topk_logprobs"]), student.gather(1, ids))
    torch.testing.assert_close(torch.tensor(scores["teacher_target_logprob"]), teacher.gather(1, target).squeeze(1))
    torch.testing.assert_close(torch.tensor(scores["student_target_logprob"]), student.gather(1, target).squeeze(1))


def test_same_prompt_scores_the_same(model):
    example = make_example()
    example["teacher_prompt"] = example["student_prompt"]
    scores = score_example(model, example, k=K)
    torch.testing.assert_close(torch.tensor(scores["student_topk_logprobs"]),
                               torch.tensor(scores["teacher_topk_logprobs"]))


def test_main_resumes(model, tmp_path, monkeypatch):
    monkeypatch.setattr("sd.score_distill.load_model", lambda *a, **kw: model)
    examples = tmp_path / "examples.jsonl"
    output = tmp_path / "scored.jsonl"
    examples.write_text("".join(json.dumps(make_example(f"t-{i}")) + "\n" for i in range(3)))
    first = make_example("t-0") | score_example(model, make_example("t-0"), k=K)
    output.write_text(json.dumps(first) + "\n" + '{"instance_id": "t-1", "trunc')  # interrupted mid-write

    main(examples, output=output, k=K, device="cpu")

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["instance_id"] for r in rows] == ["t-0", "t-1", "t-2"]
    assert all(len(r["teacher_topk_ids"]) == len(r["target"]) for r in rows)

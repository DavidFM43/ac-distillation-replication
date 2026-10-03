"""Browse offline self-distillation examples: the conversation, the hint, and the trained turn.

    uv run python -m sd.distill_viz runs/distill/examples.jsonl runs/train250/trajectories.jsonl
    # then open http://127.0.0.1:8766

Serves sd/distill_viz.html, an index of examples, and one conversation at a time (loaded on click;
the conversations are too large to put in one page). Binds to 127.0.0.1 only.
"""

import http.server
import json
from functools import partial
from pathlib import Path

import typer

from sd.distill_data import load_records

PAGE = Path(__file__).with_name("distill_viz.html")


def text_of(content) -> tuple[str, str]:
    """(thinking, visible text) from a stored message's content."""
    if isinstance(content, str):
        return "", content
    thinking = "\n".join(p.get("thinking", "") for p in content if p.get("type") == "thinking")
    text = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
    return thinking, text


def tool_calls_of(message: dict) -> list[dict]:
    calls = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"raw": fn.get("arguments")}
        calls.append({"name": fn.get("name"), "command": args.get("command", "") if isinstance(args, dict) else ""})
    return calls


def conversation(record: dict, example: dict) -> dict:
    """The trajectory as turns: each assistant turn with the outputs/messages that followed it."""
    messages = record["messages"]
    turns, preamble, turn = [], [], 0
    for m in messages:
        if m["role"] == "assistant":
            turn += 1
            thinking, text = text_of(m["content"])
            stored = record["turns"][turn - 1] if turn <= len(record["turns"]) else {}
            turns.append({"turn": turn, "thinking": thinking, "text": text, "tool_calls": tool_calls_of(m),
                          "context_len": stored.get("prompt_len"), "output_len": len(stored.get("tokens", [])),
                          "results": []})
        elif turns:  # tool outputs (or a correction message) after an assistant turn
            turns[-1]["results"].append({"role": m["role"], "content": text_of(m["content"])[1]})
        else:
            preamble.append({"role": m["role"], "content": text_of(m["content"])[1]})
    return {
        "instance_id": record["instance_id"],
        "hint_turn": example["hint_turn"],
        "n_supervised": example.get("n_supervised", 1),
        "hint": example["hint"],
        "target_len": supervised_len(example),
        "teacher_extra": len(example["teacher_prompt"]) - len(example["student_prompt"]),
        "outcome": {k: record[k] for k in ("submitted", "submit_turn", "tests_passed", "stop_reason", "n_turns")},
        "preamble": preamble,
        "turns": turns,
    }


def supervised_len(example: dict) -> int:
    """Trained tokens: the target minus tool outputs between turns (examples without a mask are all target)."""
    return sum(example.get("target_mask") or [1] * len(example["target"]))


class Handler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, index: list[dict], examples: dict, records: dict, **kwargs):
        self.index, self.examples, self.records = index, examples, records
        super().__init__(*args, **kwargs)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/index.json":
            self._send(json.dumps(self.index).encode(), "application/json")
        elif path.startswith("/example/"):
            iid = path.removeprefix("/example/")
            if iid not in self.examples:
                return self.send_error(404)
            body = conversation(self.records[iid], self.examples[iid])
            self._send(json.dumps(body).encode(), "application/json")
        else:
            self.send_error(404)

    def _send(self, body: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main(examples_path: Path, trajectories: Path, port: int = 8766) -> None:
    records = {r["instance_id"]: r for r in load_records(trajectories)}
    examples = {e["instance_id"]: e for e in map(json.loads, examples_path.read_text().splitlines())}
    index = [
        {"instance_id": iid, "hint_turn": e["hint_turn"], "n_supervised": e.get("n_supervised", 1),
         "target_len": supervised_len(e),
         "context_len": len(e["student_prompt"]),
         **{k: records[iid][k] for k in ("submitted", "submit_turn", "tests_passed", "stop_reason", "n_turns")}}
        for iid, e in examples.items()
    ]
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", port), partial(Handler, index=index, examples=examples, records=records))
    print(f"{len(index)} examples: http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    typer.run(main)

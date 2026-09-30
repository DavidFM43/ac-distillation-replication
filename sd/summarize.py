"""Summarize a trajectories.jsonl: submit behavior, turn distribution, outcomes.

    uv run python -m sd.summarize runs/baseline/trajectories.jsonl --threshold 40

Metrics follow the talk:
- task complete rate: called task_complete (by turn `threshold`)
- test pass rate:     tests pass on the final sandbox state, whether or not it submitted
- SWE-bench pass:     both
"""

import json
import math
from collections import Counter
from pathlib import Path

import typer


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion; honest at small n, unlike +-1.96*sqrt(p(1-p)/n)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (center - half, center + half)


def rate(label: str, k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{label:<36} {k:>3}/{n:<3} = {k / n if n else 0:5.0%}   95% CI [{lo:.0%}, {hi:.0%}]"


def main(path: Path, threshold: int = 40) -> None:
    # A resumed run can append a retried task again; keep the latest record per task.
    latest = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        latest[record["instance_id"]] = record
    records = list(latest.values())
    errored = [r for r in records if r["error"]]
    ok = [r for r in records if not r["error"]]
    n = len(ok)
    early = [r for r in ok if r["submitted"] and r["submit_turn"] <= threshold]
    graded = [r for r in ok if r["tests_passed"] is not None]

    print(f"{len(records)} trajectories from {path} ({len(errored)} errored, excluded below)\n")
    print(rate("submitted at all", sum(r["submitted"] for r in ok), n))
    print(rate(f"task complete by turn {threshold}", len(early), n))
    print(rate("tests pass (irrespective of submit)", sum(r["tests_passed"] for r in graded), len(graded)))
    print(rate(f"tests pass AND submitted by {threshold}", sum(bool(r["tests_passed"]) for r in early), len(graded)))

    print("\nstop reasons:", dict(Counter(r["stop_reason"] for r in ok)))
    print("turns per trajectory:", sorted(r["n_turns"] for r in ok))
    print("submit turns:", sorted(r["submit_turn"] for r in ok if r["submitted"]))
    for r in errored:
        print(f"ERROR {r['instance_id']}: {r['error']}")


if __name__ == "__main__":
    typer.run(main)

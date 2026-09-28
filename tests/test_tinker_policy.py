import pytest

from sd.tinker_policy import estimate_cost


def test_cost_estimate_counts_previous_turn_as_cached():
    turns = [{"prompt_len": 1000, "tokens": [0] * 100}, {"prompt_len": 1300, "tokens": [0] * 50}]
    # turn 1: 1000 uncached + 100 sampled; turn 2: 1100 cached, 200 uncached, 50 sampled
    expected = (1200 * 0.66 + 1100 * 0.132 + 150 * 1.995) / 1e6
    assert estimate_cost("Qwen/Qwen3.5-9B", turns) == pytest.approx(expected)

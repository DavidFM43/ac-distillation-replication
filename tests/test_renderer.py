"""Offline tests: the Qwen3.5 agent renderer against Qwen's official chat template."""

import pytest

from sd.renderer import get_agent_renderer

MODEL = "Qwen/Qwen3.5-9B"
BASH = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
        },
    },
}
# A sampled turn as the model writes it after the `<think>\n` prefill.
TURN_1 = (
    "Let me look at the repo first.\n</think>\n\nListing files.\n\n"
    "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>"
)


@pytest.fixture(scope="module")
def renderer():
    return get_agent_renderer(MODEL)


def two_turns(renderer):
    """Render turn 1, 'sample' TURN_1, add a tool result, render turn 2."""
    messages = renderer.create_conversation_prefix_with_tools([BASH], system_prompt="You are helpful.")
    messages += [{"role": "user", "content": "Fix the bug."}]
    prompt_1 = renderer.build_generation_prompt(messages).to_ints()
    completion = renderer.tokenizer.encode(TURN_1, add_special_tokens=False)
    completion.append(renderer.tokenizer.convert_tokens_to_ids("<|im_end|>"))
    assistant, _ = renderer.parse_response(completion)
    messages += [assistant, {"role": "tool", "content": "a\nb", "tool_call_id": "call_0", "name": "bash"}]
    prompt_2 = renderer.build_generation_prompt(messages).to_ints()
    return prompt_1, completion, prompt_2, assistant


def test_parses_thinking_text_and_tool_call(renderer):
    _, _, _, assistant = two_turns(renderer)
    parts = {p["type"]: p for p in assistant["content"]}
    assert parts["thinking"]["thinking"] == "Let me look at the repo first."
    assert parts["text"]["text"] == "Listing files."  # trimmed, as the template does
    assert assistant["tool_calls"][0].function.name == "bash"


def test_next_prompt_extends_previous_prompt_and_completion(renderer):
    """Token-exact history: turn-2 prompt starts with turn-1 prompt + turn-1 sampled tokens."""
    prompt_1, completion, prompt_2, _ = two_turns(renderer)
    prefix = prompt_1 + completion
    assert prompt_2[: len(prefix)] == prefix


def test_prompt_matches_official_chat_template(renderer):
    """Thinking is kept in agent-loop history, as Qwen's HF chat template does."""
    _, _, prompt_2, _ = two_turns(renderer)
    hf_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Fix the bug."},
        {
            "role": "assistant",
            "reasoning_content": "Let me look at the repo first.",
            "content": "Listing files.",
            "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": {"command": "ls -la"}}}],
        },
        {"role": "tool", "content": "a\nb"},
    ]
    expected = renderer.tokenizer.apply_chat_template(hf_messages, tools=[BASH], tokenize=False, add_generation_prompt=True)
    assert renderer.tokenizer.decode(prompt_2) == expected


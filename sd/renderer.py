"""Qwen3.5 renderer that reproduces Qwen's official chat template in agent loops.

Two differences from tinker_cookbook's default Qwen3.5 renderer:

1. Thinking is kept in history. The official template keeps the <think> block of every
   assistant turn after the last real user message (tool results don't count), i.e. every
   turn of an agent loop. The cookbook default strips it.
2. Text around thinking and tool calls is trimmed after parsing, as the template does
   (`content|trim`). Otherwise re-rendering history adds the renderer's separator on top of
   the whitespace kept by the parser, and the prompt drifts from the official template.

With both, each turn's prompt is exactly the previous prompt + the sampled tokens + the
tool results, which is what later token-level re-scoring (teacher/student) relies on.
"""

from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import ParseTermination
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer


class Qwen35AgentRenderer(Qwen3_5Renderer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, strip_thinking_from_history=False)

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        message, termination = super().parse_response(response)
        message["content"] = trim_text(message["content"])
        return message, termination


def trim_text(content):
    if isinstance(content, str):
        return content.strip()
    trimmed = []
    for part in content:
        if part["type"] == "text":
            part = {**part, "text": part["text"].strip()}
            if not part["text"]:
                continue
        trimmed.append(part)
    return trimmed


def get_agent_renderer(model_name: str) -> Qwen35AgentRenderer:
    return Qwen35AgentRenderer(tokenizer_utils.get_tokenizer(model_name))

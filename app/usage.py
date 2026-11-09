"""Token usage reported by LLM APIs, in one shape for every provider.

Providers count differently: Claude's input_tokens leaves out the tokens read from or written to its prompt cache,
OpenAI's prompt_tokens includes cached tokens, and reasoning models bill thinking as output. Usage counts every input
token in input and every output token in output, and says how many of them were cached or reasoning. Relays send
counts as numbers, numeric strings, null or not at all; reported tells "the API said nothing" from real zeros.
"""

import math
from dataclasses import dataclass


@dataclass
class Usage:
    input: int = 0  # all input tokens, those read from or written to a prompt cache included
    output: int = 0  # all output tokens, reasoning included
    reasoning: int = 0  # of output: thinking / reasoning tokens
    cached_input: int = 0  # of input: read from the provider's prompt cache
    cache_write: int = 0  # of input: written to the prompt cache (Claude)
    reported: bool = False  # the API reported usage at all; without it the counts above say nothing

    def add(self, other: "Usage") -> None:
        self.input += other.input
        self.output += other.output
        self.reasoning += other.reasoning
        self.cached_input += other.cached_input
        self.cache_write += other.cache_write
        self.reported = self.reported or other.reported


def count(value: object) -> int | None:
    """A token count as an API sent it: a number or a numeric string. None for anything else (missing, null,
    negative, not a number), which counts as not reported."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value) and value >= 0:
        return int(value)
    return None


def nested(data: object, *path: str) -> int | None:
    """count() of data[path[0]][path[1]]..., None when any level is missing or not an object."""
    for key in path:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return count(data)


def reported(
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    reasoning: int | None = None,
    cached_input: int | None = None,
    cache_write: int | None = None,
) -> Usage:
    """Usage from the counts a reply carried. It counts as reported when the reply gave input or output tokens; the
    parts it left out count as 0."""
    if input_tokens is None and output_tokens is None:
        return Usage()
    return Usage(
        input=input_tokens or 0,
        output=output_tokens or 0,
        reasoning=reasoning or 0,
        cached_input=cached_input or 0,
        cache_write=cache_write or 0,
        reported=True,
    )

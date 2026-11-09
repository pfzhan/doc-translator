"""Inline code kept as the source has it.

Inline code, keys, sample output and variables go to the translation service inside their paragraph, as context,
and models sometimes translate or retype them. Each one in the translation is matched with one of the source's
and gets the source's text back. The request and the cache stay as they are: cached translations are restored the
same way when the output is written.
"""

from collections.abc import Sequence


def match_verbatim(sources: Sequence[str], outputs: Sequence[str]) -> list[int | None]:
    """For each verbatim span of the translation (outputs), the index of the source span to restore it from, or
    None to leave it as the model wrote it.

    A span the model left unchanged is matched with an equal one of the source, wherever the model moved it. The
    others are matched in order, but only when as many are left on both sides: when the model dropped or added a
    span, which of them it changed cannot be told, and they are left alone. A duplicate of a span already matched
    keeps its source's text too.
    """
    result: list[int | None] = [None] * len(outputs)
    unused = list(range(len(sources)))
    for i, out in enumerate(outputs):
        for j in unused:
            if sources[j] == out:
                result[i] = j
                unused.remove(j)
                break
    rest = [i for i, j in enumerate(result) if j is None]
    if len(rest) == len(unused):
        for i, j in zip(rest, unused):
            result[i] = j
    else:
        for i in rest:
            if outputs[i] in sources:
                result[i] = sources.index(outputs[i])
    return result

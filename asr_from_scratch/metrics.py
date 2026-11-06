"""Word Error Rate (WER) and Character Error Rate (CER)."""
from __future__ import annotations


def _edit_distance(a: list, b: list) -> int:
    """Levenshtein distance: minimum number of insertions, deletions and substitutions."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def wer(references: list[str], hypotheses: list[str]) -> float:
    errors = sum(_edit_distance(r.split(), h.split()) for r, h in zip(references, hypotheses))
    return errors / max(1, sum(len(r.split()) for r in references))


def cer(references: list[str], hypotheses: list[str]) -> float:
    errors = sum(_edit_distance(list(r), list(h)) for r, h in zip(references, hypotheses))
    return errors / max(1, sum(len(r) for r in references))

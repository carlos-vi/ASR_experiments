"""Word n-gram language model in ARPA format.

CTC decides each character independently, so it has no notion of which words
exist or which word sequences are likely. An n-gram model supplies that knowledge
at decoding time, at no training cost: ``P(word | previous words)`` with the
standard Katz backoff to shorter contexts.
"""
from __future__ import annotations

import gzip
import math
from collections.abc import Sequence
from pathlib import Path

LOG10_TO_LN = math.log(10.0)
# Unknown words must score well below any sequence of known words. The <unk> entry
# of a corpus LM is far too generous for this: since CTC can spell anything, the
# decoder would otherwise glue words together to obtain cheap unknown tokens.
UNKNOWN_LOG10 = -15.0


class ArpaLanguageModel:
    def __init__(self, path: str | Path) -> None:
        self.grams: dict[tuple[str, ...], tuple[float, float]] = {}
        self.order = 1
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as file:
            size = 0
            for line in file:
                line = line.strip()
                if not line or line == "\\data\\" or line == "\\end\\" or line.startswith("ngram "):
                    continue
                if line.startswith("\\") and line.endswith("-grams:"):
                    size = int(line[1:line.index("-")])
                    self.order = max(self.order, size)
                    continue
                if size == 0:
                    continue
                fields = line.split()
                words = tuple(fields[1:size + 1])
                backoff = float(fields[size + 1]) if len(fields) > size + 1 else 0.0
                # Sentence markers keep their case; corpus words are usually uppercase.
                words = tuple(w if w.startswith("<") else w.lower() for w in words)
                self.grams[words] = (float(fields[0]), backoff)
        self.word_prefixes = {word[:end] for (word, *rest) in self.grams if not rest
                              for end in range(1, len(word) + 1)}

    def is_word_prefix(self, partial: str) -> bool:
        """Whether a partially spelled word can still become a known word."""
        return not partial or partial in self.word_prefixes

    def score(self, history: Sequence[str], word: str) -> float:
        """Natural-log probability of ``word`` given the preceding words."""
        context = tuple(history[-(self.order - 1):]) if self.order > 1 else ()
        backoff = 0.0
        while True:
            entry = self.grams.get(context + (word,))
            if entry is not None:
                return (backoff + entry[0]) * LOG10_TO_LN
            if not context:
                return (backoff + UNKNOWN_LOG10) * LOG10_TO_LN
            parent = self.grams.get(context)
            if parent is not None:
                backoff += parent[1]
            context = context[1:]

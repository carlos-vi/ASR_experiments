"""CTC decoding: greedy search and prefix beam search with a language model."""
from __future__ import annotations

import heapq
import math

import numpy as np
import torch

from .language_model import LOG10_TO_LN, UNKNOWN_LOG10, ArpaLanguageModel
from .vocabulary import VOCABULARY, decode

_LOG_ZERO = -math.inf
_SPACE = " "


def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor) -> list[str]:
    """Pick the most likely symbol per frame, collapse repeats and remove blanks.

    Example: [h h <b> e l <b> l o o] -> "hello". The blank between the two l's is
    what makes double letters possible.
    """
    best = log_probs.argmax(-1)
    texts = []
    for row, length in zip(best, lengths):
        ids, previous = [], -1
        for index in row[: int(length)].tolist():
            if index != previous:
                ids.append(index)
            previous = index
        texts.append(decode(ids))
    return texts


def _log_add(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    if b == _LOG_ZERO:
        return a
    return a + math.log1p(math.exp(b - a))


def _word_score(prefix: str, lm: ArpaLanguageModel, alpha: float, beta: float) -> float:
    """Language model contribution of the last complete word in ``prefix``."""
    words = prefix.split()
    return alpha * lm.score(["<s>"] + words[:-1], words[-1]) + beta


def beam_search_decode(
    log_probs: np.ndarray,
    lm: ArpaLanguageModel | None = None,
    alpha: float = 0.7,
    beta: float = 1.0,
    beam_width: int = 64,
    prune_below: float = -6.0,
) -> str:
    """CTC prefix beam search (Hannun et al., 2014) over a (frames, vocab) array.

    Each hypothesis tracks two probabilities: paths ending in a blank and paths
    ending in a character, because only the former can extend with a repeated
    letter. When a word is completed, the language model score is added:
    ``alpha`` weights the language model and ``beta`` rewards each word, which
    counteracts the tendency to prefer fewer words. While a word is being spelled
    it has no language model score yet, so hypotheses are penalized as soon as the
    partial word cannot become a known word; otherwise skipping spaces would be
    the cheapest option. Symbols with a log-probability under ``prune_below`` are
    skipped: most frames are confidently blank, so this is what keeps a pure
    Python search practical.
    """
    beams: dict[str, tuple[float, float]] = {"": (0.0, _LOG_ZERO)}
    lm_scores: dict[str, float] = {"": 0.0}
    unknown_penalty = alpha * UNKNOWN_LOG10 * LOG10_TO_LN

    def rank(item: tuple[str, list[float]]) -> float:
        prefix, (p_blank, p_char) = item
        score = _log_add(p_blank, p_char) + lm_scores[prefix]
        if lm is not None and not lm.is_word_prefix(prefix[prefix.rfind(_SPACE) + 1:]):
            score += unknown_penalty
        return score

    for frame in log_probs:
        candidates = np.flatnonzero(frame >= prune_below).tolist()
        values = frame[candidates].tolist()
        next_beams: dict[str, list[float]] = {}

        def add(prefix: str, ends_in_blank: float, ends_in_char: float) -> None:
            entry = next_beams.setdefault(prefix, [_LOG_ZERO, _LOG_ZERO])
            entry[0] = _log_add(entry[0], ends_in_blank)
            entry[1] = _log_add(entry[1], ends_in_char)

        for prefix, (p_blank, p_char) in beams.items():
            total = _log_add(p_blank, p_char)
            for token, value in zip(candidates, values):
                if token == 0:
                    add(prefix, total + value, _LOG_ZERO)
                    continue
                char = VOCABULARY[token]
                if char == _SPACE and (not prefix or prefix[-1] == _SPACE):
                    # Leading and double spaces carry no information.
                    add(prefix, total + value, _LOG_ZERO)
                    continue
                if prefix and prefix[-1] == char:
                    add(prefix, _LOG_ZERO, p_char + value)  # repeat collapses
                    extended, source = prefix + char, p_blank + value
                else:
                    extended, source = prefix + char, total + value
                if source > _LOG_ZERO:
                    add(extended, _LOG_ZERO, source)
                    if extended not in lm_scores:
                        score = lm_scores[prefix]
                        if lm is not None and char == _SPACE:
                            score += _word_score(extended, lm, alpha, beta)
                        lm_scores[extended] = score

        ranked = heapq.nlargest(beam_width, next_beams.items(), key=rank)
        beams = {prefix: (scores[0], scores[1]) for prefix, scores in ranked}
        lm_scores = {prefix: lm_scores[prefix] for prefix in beams}

    def final_score(prefix: str) -> float:
        score = _log_add(*beams[prefix]) + lm_scores[prefix]
        if lm is not None:
            if prefix and not prefix.endswith(_SPACE):
                score += _word_score(prefix, lm, alpha, beta)
            score += alpha * lm.score(["<s>"] + prefix.split(), "</s>")
        return score

    return max(beams, key=final_score).strip()

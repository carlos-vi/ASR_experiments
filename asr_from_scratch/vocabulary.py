"""Character-level tokenizer for English.

The model predicts characters rather than words. The vocabulary stays tiny and
out-of-vocabulary words do not exist. CTC also requires a special *blank*
symbol, which takes index 0.
"""
from __future__ import annotations

import re
import unicodedata

BLANK = "<blank>"
CHARACTERS = " abcdefghijklmnopqrstuvwxyz'"
VOCABULARY = [BLANK] + list(CHARACTERS)
CHAR_TO_ID = {char: index for index, char in enumerate(VOCABULARY)}
VOCAB_SIZE = len(VOCABULARY)


def normalize(text: str) -> str:
    """Lowercase ASCII letters, in-word apostrophes and single spaces.

    Accents are stripped, and digits and other symbols are removed: transcribing
    numbers would require spelling them out ("42" -> "forty two"), which is out
    of scope for this model.
    """
    text = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = "".join(char if char in CHAR_TO_ID else " " for char in text)
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def encode(text: str) -> list[int]:
    return [CHAR_TO_ID[char] for char in normalize(text)]


def decode(ids: list[int]) -> str:
    return "".join(VOCABULARY[index] for index in ids if index != 0)

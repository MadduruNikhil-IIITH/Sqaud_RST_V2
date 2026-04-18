from __future__ import annotations

from typing import List, Tuple


def naive_sentence_split(text: str) -> List[str]:
    parts: list[str] = []
    start = 0
    for idx, ch in enumerate(text):
        if ch in {".", "?", "!"}:
            piece = text[start : idx + 1].strip()
            if piece:
                parts.append(piece)
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def locate_sentences_with_offsets(paragraph: str, sentences: List[str]) -> List[Tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for sentence in sentences:
        pos = paragraph.find(sentence, cursor)
        if pos == -1:
            raise ValueError(f"Unable to align sentence: {sentence[:60]}")
        start = pos
        end = pos + len(sentence)
        offsets.append((start, end))
        cursor = end
    return offsets

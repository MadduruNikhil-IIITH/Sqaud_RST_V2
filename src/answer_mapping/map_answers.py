from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.common.constants import ANSWER_ID_TEMPLATE
from src.common.io_utils import load_json, write_jsonl


def _load_sentence_ranges(sentence_csv: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    with sentence_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            para_id = row["para_id"]
            grouped.setdefault(para_id, []).append(
                {
                    "sent_id": row["sent_id"],
                    "start": int(row["start_char"]),
                    "end": int(row["end_char"]),
                }
            )
    return grouped


def map_answers(sample_manifest_path: Path, sentence_csv: Path, output_jsonl: Path) -> list[dict[str, Any]]:
    manifest = load_json(sample_manifest_path)
    sentence_ranges = _load_sentence_ranges(sentence_csv)
    mappings: list[dict[str, Any]] = []

    for paragraph in manifest["paragraphs"]:
        para_id = paragraph["para_id"]
        ranges = sentence_ranges.get(para_id, [])

        for q in paragraph.get("qas", []):
            q_id = q.get("id", "unknown")
            answers = q.get("answers", {})
            if not answers and q.get("plausible_answers"):
                answers = q.get("plausible_answers", {})

            # Handle SQuAD v2.0 format where answers is a dict with lists
            if isinstance(answers, dict) and "text" in answers:
                answer_texts = answers.get("text", [])
                answer_starts = answers.get("answer_start", [])
                answers_list = [
                    {"text": text, "answer_start": start}
                    for text, start in zip(answer_texts, answer_starts)
                ]
            else:
                answers_list = answers if isinstance(answers, list) else []

            for ans_idx, answer in enumerate(answers_list):
                answer_start = int(answer.get("answer_start", -1))
                answer_text = answer.get("text", "")
                answer_end = answer_start + len(answer_text)
                mapped_sent_ids: list[str] = []

                for sentence in ranges:
                    overlap = answer_start < sentence["end"] and answer_end > sentence["start"]
                    if overlap:
                        mapped_sent_ids.append(sentence["sent_id"])

                mappings.append(
                    {
                        "para_id": para_id,
                        "question_id": q_id,
                        "answer_id": ANSWER_ID_TEMPLATE.format(question_id=q_id, answer_idx=ans_idx),
                        "answer_start": answer_start,
                        "answer_end": answer_end,
                        "answer_text": answer_text,
                        "mapped_sent_ids": mapped_sent_ids,
                        "status": "ok" if mapped_sent_ids else "unmapped",
                    }
                )

    write_jsonl(output_jsonl, mappings)
    return mappings

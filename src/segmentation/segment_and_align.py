from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.constants import SENT_ID_TEMPLATE
from src.common.io_utils import load_json, write_csv, write_json
from src.common.text_utils import locate_sentences_with_offsets, naive_sentence_split


def segment_and_align(sample_manifest_path: Path, sentence_table_path: Path, diagnostics_path: Path) -> dict[str, Any]:
    manifest = load_json(sample_manifest_path)

    sentence_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for paragraph in manifest["paragraphs"]:
        para_id = paragraph["para_id"]
        context = paragraph["context"]
        q_ids = [q.get("id") for q in paragraph.get("qas", [])]

        sentences = naive_sentence_split(context)
        offsets = locate_sentences_with_offsets(context, sentences)

        for sent_idx, (sent_text, (start, end)) in enumerate(zip(sentences, offsets)):
            reconstructed = context[start:end]
            aligned = reconstructed == sent_text
            sent_id = SENT_ID_TEMPLATE.format(para_id=para_id, sent_idx=sent_idx)

            sentence_rows.append(
                {
                    "sent_id": sent_id,
                    "para_id": para_id,
                    "sent_idx": sent_idx,
                    "sent_text": sent_text,
                    "start_char": start,
                    "end_char": end,
                    "question_ids": "|".join([qid for qid in q_ids if qid]),
                }
            )
            diagnostics.append(
                {
                    "sent_id": sent_id,
                    "para_id": para_id,
                    "is_aligned": aligned,
                    "reconstructed": reconstructed,
                }
            )

    if not all(item["is_aligned"] for item in diagnostics):
        failed = [d for d in diagnostics if not d["is_aligned"]]
        write_json(diagnostics_path, {"status": "failed", "failed_count": len(failed), "items": failed})
        raise ValueError("Offset alignment failed; inspect diagnostics")

    write_csv(
        sentence_table_path,
        sentence_rows,
        fieldnames=[
            "sent_id",
            "para_id",
            "sent_idx",
            "sent_text",
            "start_char",
            "end_char",
            "question_ids",
        ],
    )
    write_json(diagnostics_path, {"status": "ok", "count": len(diagnostics)})
    return {"status": "ok", "sentence_count": len(sentence_rows)}

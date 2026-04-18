from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.common.models import GoldSalienceRecord
from src.common.io_utils import write_csv


def build_gold_labels(sentence_csv: Path, mapping_jsonl: Path, output_csv: Path) -> list[dict[str, Any]]:
    salient_sentences: set[str] = set()
    with mapping_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for sent_id in row.get("mapped_sent_ids", []):
                salient_sentences.add(sent_id)

    gold_rows: list[dict[str, Any]] = []
    with sentence_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sent_id = row["sent_id"]
            record = GoldSalienceRecord(
                sent_id=sent_id,
                para_id=row["para_id"],
                sent_idx=int(row["sent_idx"]),
                gold_salient=1 if sent_id in salient_sentences else 0,
            )
            gold_rows.append(record.to_dict())

    write_csv(
        output_csv,
        gold_rows,
        ["sent_id", "para_id", "sent_idx", "gold_salient", "gold_rank_optional"],
    )
    return gold_rows

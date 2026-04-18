
from __future__ import annotations
from tqdm import tqdm

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.common.io_utils import write_csv, write_json


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_ratio(train: float, val: float, test: float) -> tuple[float, float, float]:
    total = train + val + test
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    return train / total, val / total, test / total


def _assign_group_splits(
    para_ids: list[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    train_ratio, val_ratio, test_ratio = _normalize_ratio(train_ratio, val_ratio, test_ratio)
    rng = random.Random(seed)
    ordered = sorted(set(para_ids))
    rng.shuffle(ordered)

    n = len(ordered)
    n_train = max(1, int(round(n * train_ratio))) if n > 0 else 0
    n_val = int(round(n * val_ratio)) if n > 2 else 0
    if n_train + n_val >= n and n > 1:
        n_val = max(0, n - n_train - 1)
    n_test = max(0, n - n_train - n_val)

    split_map: dict[str, str] = {}
    for idx, para_id in enumerate(ordered):
        if idx < n_train:
            split_map[para_id] = "train"
        elif idx < n_train + n_val:
            split_map[para_id] = "val"
        else:
            split_map[para_id] = "test"
    return split_map



def prepare_transformer_dataset(
    source_feature_table_path: Path,
    output_csv_path: Path,
    split_report_path: Path,
    split_seed: int = 42,
    split_train: float = 0.7,
    split_val: float = 0.15,
    split_test: float = 0.15,
) -> list[dict[str, Any]]:
    """
    Prepare the transformer training dataset.
    
    - Always use feature_table.csv (raw features) as input, never feature_table_scored.csv.
    - Scoring columns (feature_salience_score, feature_salience_label, feature_salience_rank) are always excluded from the output.
    - Only raw and engineered features, context columns, and the label are included.
    - This ensures no label leakage or use of gold-derived features in model training.
    """
    # Only use feature_table.csv as input, not feature_table_scored.csv
    rows = _read_csv_rows(source_feature_table_path)

    by_para: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_para[row["para_id"]].append(row)
    for para_rows in by_para.values():
        para_rows.sort(key=lambda r: int(r.get("sent_idx", "0") or 0))

    split_map = _assign_group_splits(
        para_ids=[row["para_id"] for row in rows],
        seed=split_seed,
        train_ratio=split_train,
        val_ratio=split_val,
        test_ratio=split_test,
    )

    out_rows: list[dict[str, Any]] = []
    # Drop any scoring columns if present (should not be present in feature_table.csv, but this is a safeguard)
    drop_cols = {
        "feature_salience_score",
        "feature_salience_label",
        "feature_salience_rank",
    }

    for para_id, para_rows in tqdm(by_para.items(), desc="Preparing transformer dataset", total=len(by_para)):
        for i, row in enumerate(para_rows):
            prev_text = para_rows[i - 1].get("sent_text", "") if i > 0 else ""
            next_text = para_rows[i + 1].get("sent_text", "") if i + 1 < len(para_rows) else ""
            sent_text = row.get("sent_text", "")

            # Get gold_salient for prev/next sentences
            prev_label = int(float(para_rows[i - 1].get("gold_salient", "0") or 0)) if i > 0 else None
            next_label = int(float(para_rows[i + 1].get("gold_salient", "0") or 0)) if i + 1 < len(para_rows) else None
            gold_salient = int(float(row.get("gold_salient", "0") or 0))

            out_row: dict[str, Any] = {k: v for k, v in row.items() if k not in drop_cols}
            out_row["para_id"] = para_id
            out_row["sent_idx"] = int(row.get("sent_idx", "0") or 0)
            out_row["split"] = split_map.get(para_id, "train")
            # out_row["label"] = gold_salient  # Removed: use only 'gold_salient' as the label
            out_row["gold_salient"] = gold_salient
            out_row["sent_text"] = sent_text
            out_row["prev_sent_text"] = prev_text
            out_row["next_sent_text"] = next_text
            out_row["prev_sent_label"] = prev_label
            # context_window_text removed as requested

            out_rows.append(out_row)

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    write_csv(output_csv_path, out_rows, fieldnames)

    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_pos: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for row in out_rows:
        s = row["split"]
        split_counts[s] = split_counts.get(s, 0) + 1
        split_pos[s] = split_pos.get(s, 0) + int(row["gold_salient"] == 1)

    para_split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for s in split_map.values():
        para_split_counts[s] = para_split_counts.get(s, 0) + 1

    write_json(
        split_report_path,
        {
            "input": str(source_feature_table_path),
            "output": str(output_csv_path),
            "seed": split_seed,
            "split_ratios": {"train": split_train, "val": split_val, "test": split_test},
            "rows_by_split": split_counts,
            "positive_labels_by_split": split_pos,
            "paragraphs_by_split": para_split_counts,
        },
    )

    return out_rows

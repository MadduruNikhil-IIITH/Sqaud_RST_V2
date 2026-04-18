from __future__ import annotations

import csv
import math
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from typing import Any

from src.common.io_utils import write_csv, write_json


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        num = float(text)
    except ValueError:
        return None
    if math.isnan(num):
        return None
    return num


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _zscore_values(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    vals: list[float] = []
    for row in rows:
        v = _to_float(row.get(field))
        if v is not None:
            vals.append(v)
    if not vals:
        return {}

    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    std = math.sqrt(var)
    if std == 0:
        return {row["sent_id"]: 0.0 for row in rows}

    out: dict[str, float] = {}
    for row in rows:
        v = _to_float(row.get(field))
        out[row["sent_id"]] = 0.0 if v is None else (v - mean) / std
    return out


def _rank_desc(scores: list[tuple[str, float]]) -> dict[str, int]:
    ordered = sorted(scores, key=lambda x: x[1], reverse=True)
    ranks: dict[str, int] = {}
    rank = 1
    last_score = None
    for idx, (sent_id, score) in enumerate(ordered):
        if last_score is None or score != last_score:
            rank = idx + 1
            last_score = score
        ranks[sent_id] = rank
    return ranks


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def _classification_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if _to_float(r.get("gold_salient")) is not None]
    if not valid:
        return {"has_gold": False, "row_count": len(rows)}

    tp = fp = tn = fn = 0
    for r in valid:
        y_true = 1 if _safe_int(r.get("gold_salient"), 0) == 1 else 0
        y_pred = 1 if _safe_int(r.get("feature_salience_label"), 0) == 1 else 0
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(valid) if valid else 0.0

    return {
        "has_gold": True,
        "row_count": len(rows),
        "valid_gold_rows": len(valid),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def score_feature_salience(
    feature_table_path: Path,
    output_csv_path: Path,
    report_json_path: Path,
) -> list[dict[str, Any]]:
    rows = _read_csv_rows(feature_table_path)

    # Weighted z-score blend for first-pass salience scoring.
    weights: dict[str, float] = {
        "span_importance_score": 0.20,
        "rst_tree_depth": -0.10,
        "sentence_position_ratio": 0.05,
        "named_entity_count": 0.10,
        "prev_next_cohesion_score": 0.10,
        "paragraph_discourse_continuity_score": 0.10,
        "content_word_density": 0.10,
        "lexical_density": 0.05,
        "surprisal_sentence_per_token": 0.10,
        "surprisal_word_mean": 0.05,
        "surprisal_word_max": 0.05,
    }

    zmaps: dict[str, dict[str, float]] = {
        field: _zscore_values(rows, field) for field in weights
    }

    for row in rows:
        sent_id = row["sent_id"]
        score = 0.0
        for field, w in weights.items():
            score += w * zmaps[field].get(sent_id, 0.0)
        row["feature_salience_score"] = score

    by_para: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_para[row["para_id"]].append(row)

    for para_id, para_rows in tqdm(by_para.items(), desc='Scoring salience (paragraphs)'):
        sent_scores = [(r["sent_id"], float(r["feature_salience_score"])) for r in para_rows]
        rank_map = _rank_desc(sent_scores)
        positive_count = sum(1 for r in para_rows if _safe_int(r.get("gold_salient"), 0) == 1)
        k = positive_count if positive_count > 0 else 1

        ordered_ids = [sid for sid, _ in sorted(sent_scores, key=lambda x: x[1], reverse=True)]
        top_ids = set(ordered_ids[:k])

        for row in tqdm(para_rows, desc=f'Para {para_id} sents', leave=False):
            sid = row["sent_id"]
            row["feature_salience_rank"] = rank_map[sid]
            row["feature_salience_label"] = 1 if sid in top_ids else 0

    report = {
        "feature_table_input": str(feature_table_path),
        "feature_table_output": str(output_csv_path),
        "weights": weights,
        "metrics": _classification_report(rows),
    }

    # Ensure prev_sent_label and next_sent_label are present in output if present in input
    if rows and ("prev_sent_label" not in rows[0] or "next_sent_label" not in rows[0]):
        for row in rows:
            row["prev_sent_label"] = None
            row["next_sent_label"] = None
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(output_csv_path, rows, fieldnames)
    write_json(report_json_path, report)
    return rows

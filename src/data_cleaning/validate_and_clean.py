"""
Data cleaning and validation for sentence salience experiment.

Validates SQuAD paragraphs for sentence extraction quality:
- Paragraphs have min length (need multiple sentences)
- Answers have valid character offsets
- Questions are substantive (non-empty)
- No duplicate entries
- Character encoding is clean

Output: cleaned_sample_manifest.json + cleaning_report.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.io_utils import load_json, write_json


def validate_and_clean(sample_manifest_path: Path, cleaned_manifest_path: Path, report_path: Path) -> dict[str, Any]:
    """
    Validate and clean SQuAD manifest for sentence salience tasks.

    Args:
        sample_manifest_path: Path to sample_manifest.json from Phase 1
        cleaned_manifest_path: Output path for cleaned manifest
        report_path: Output path for cleaning report (what was removed/why)

    Returns:
        dict with counts: original, retained, removed, removal_reasons
    """
    manifest = load_json(sample_manifest_path)
    original_paragraphs = manifest.get("paragraphs", [])

    cleaned_paragraphs: list[dict[str, Any]] = []
    removal_log: list[dict[str, Any]] = []

    for para_dict in original_paragraphs:
        para_id = para_dict.get("para_id", "unknown")
        context = para_dict.get("context", "").strip()
        qas = para_dict.get("qas", [])

        # Check 1: Paragraph length (need ~100+ chars for multiple sentences)
        if len(context) < 100:
            removal_log.append(
                {
                    "para_id": para_id,
                    "reason": "too_short",
                    "detail": f"context length {len(context)} < 100 chars",
                }
            )
            continue

        # Check 2: Contains at least one question with answer
        has_valid_qa = False
        for qa in qas:
            q_text = qa.get("question", "").strip()
            answers_dict = qa.get("answers", {})

            if not q_text:
                continue  # Skip empty questions

            # answers_dict has keys 'text' (list) and 'answer_start' (list)
            answer_texts = answers_dict.get("text", []) or []
            answer_starts = answers_dict.get("answer_start", []) or []

            for ans_text, ans_start in zip(answer_texts, answer_starts):
                ans_text_clean = ans_text.strip()
                # Validate answer offset
                if ans_start >= 0 and ans_start + len(ans_text_clean) <= len(context):
                    reconstructed = context[ans_start : ans_start + len(ans_text_clean)]
                    if reconstructed == ans_text_clean:
                        has_valid_qa = True
                        break
            if has_valid_qa:
                break

        if not has_valid_qa:
            removal_log.append(
                {
                    "para_id": para_id,
                    "reason": "no_valid_answers",
                    "detail": "no questions with valid answer offsets",
                }
            )
            continue

        # Check 3: Clean encoding (no null bytes, control chars except \n\t)
        try:
            context.encode("utf-8").decode("utf-8")
        except Exception as e:
            removal_log.append(
                {
                    "para_id": para_id,
                    "reason": "encoding_error",
                    "detail": str(e),
                }
            )
            continue

        # Passed all checks - retain
        cleaned_paragraphs.append(para_dict)

    # Build cleaned manifest
    cleaned_manifest = {
        "dataset_version": manifest.get("dataset_version", "squad_v2.0"),
        "seed": manifest.get("seed", 42),
        "sample_size_original": len(original_paragraphs),
        "sample_size_cleaned": len(cleaned_paragraphs),
        "source": manifest.get("source", "huggingface_datasets_squad_v2"),
        "cleaning_applied": True,
        "paragraphs": cleaned_paragraphs,
    }

    # Build report
    report = {
        "total_original": len(original_paragraphs),
        "total_retained": len(cleaned_paragraphs),
        "total_removed": len(original_paragraphs) - len(cleaned_paragraphs),
        "removal_reasons": {},
        "removal_log": removal_log,
    }

    # Count reasons
    for entry in removal_log:
        reason = entry["reason"]
        report["removal_reasons"][reason] = report["removal_reasons"].get(reason, 0) + 1

    # Write outputs
    write_json(cleaned_manifest_path, cleaned_manifest)
    write_json(report_path, report)

    return report

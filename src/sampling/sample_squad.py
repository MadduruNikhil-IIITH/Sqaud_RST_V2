from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

from src.common.constants import DATASET_VERSION, SAMPLE_SIZE, SEED
from src.common.io_utils import write_json


def sample_paragraphs(output_path: Path, sample_size: int = SAMPLE_SIZE, seed: int = SEED) -> dict[str, Any]:
    """
    Load SQuAD v2.0 from HuggingFace datasets and sample deterministically.
    Ignores input_path parameter (kept for compatibility).
    """
    # Load SQuAD v2.0 from HuggingFace datasets (auto-downloads if needed)
    print(f"[DEBUG] sample_paragraphs called with sample_size={sample_size}, seed={seed}, output_path={output_path}")
    dataset = load_dataset("squad_v2", split="train")
    
    rows: list[dict[str, Any]] = []
    
    for example in dataset:
        para_id = f"{DATASET_VERSION}_{example['id']}"
        context = example.get("context", "")
        title = example.get("title", "unknown")
        
        # Reconstruct qas list from example structure
        # SQuAD v2.0 has: id, title, context, question, id (question id), answers
        qas_entry = {
            "id": example.get("id", "unknown"),
            "question": example.get("question", ""),
            "answers": example.get("answers", []),
            "is_impossible": example.get("is_impossible", False),
        }
        
        rows.append(
            {
                "para_id": para_id,
                "title": title,
                "context": context,
                "qas": [qas_entry],
            }
        )
    
    rng = random.Random(seed)
    if sample_size > len(rows):
        raise ValueError(f"Requested {sample_size} paragraphs but only {len(rows)} available")
    
    sampled = rng.sample(rows, sample_size)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "seed": seed,
        "sample_size": sample_size,
        "source": "huggingface_datasets_squad_v2",
        "paragraphs": sampled,
    }
    print(f"[DEBUG] Writing manifest with {sample_size} paragraphs to {output_path}")
    write_json(output_path, manifest)
    print(f"[DEBUG] Write complete. File exists: {output_path.exists()}")
    return manifest

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(stage: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.pipeline.run_stage",
        "--stage",
        stage,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    for stage in [
        "sample_dataset",
        "clean_data",
        "segment_sentences",
        "build_gold_labels",
        "parse_rst_trees",
        "feature_extraction",
        "score_feature_salience",
    ]:
        run(stage)


if __name__ == "__main__":
    main()

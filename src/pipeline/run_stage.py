from __future__ import annotations

import argparse
from pathlib import Path

from src.answer_mapping.map_answers import map_answers
from src.data_cleaning.validate_and_clean import validate_and_clean
from src.features.extract_features import extract_features
from src.gold_labels.build_gold import build_gold_labels
from src.modeling.prepare_dataset import prepare_transformer_dataset
from src.rst_parsing.parse_rst import build_rst_image_manifest, parse_rst_placeholder
from src.scoring.score_features import score_feature_salience
from src.sampling.sample_squad import sample_paragraphs
from src.segmentation.segment_and_align import segment_and_align


ALIASES = {
    "phase1": "sample_dataset",
    "phase1b": "clean_data",
    "phase2": "segment_sentences",
    "phase3": "build_gold_labels",
    "phase4": "parse_rst_trees",
    "phase5": "feature_extraction",
    "phase6": "score_feature_salience",
    "phase7prep": "prepare_transformer_dataset",
    "salience_classifier": "salience_classifier",
    "llm_classifier": "llm_classifier",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single pipeline stage")
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "sample_dataset",
            "clean_data",
            "segment_sentences",
            "build_gold_labels",
            "parse_rst_trees",
            "feature_extraction",
            "score_feature_salience",
            "prepare_transformer_dataset",
            "salience_classifier",
            "llm_classifier",
            "phase1",
            "phase1b",
            "phase2",
            "phase3",
            "phase4",
            "phase5",
            "phase6",
            "phase7prep",
        ],
    )
    parser.add_argument("--input", default="data/raw/train-v2.0.json")
    parser.add_argument("--sample-manifest", default="data/interim/sample_manifest.json")
    parser.add_argument("--cleaned-manifest", default="data/interim/cleaned_sample_manifest.json")
    parser.add_argument("--cleaning-report", default="data/interim/cleaning_report.json")
    parser.add_argument("--sentence-table", default="data/processed/sentence_table.csv")
    parser.add_argument("--segment-diagnostics", default="data/interim/segmentation_diagnostics.json")
    parser.add_argument("--answer-mapping", default="data/processed/answer_sentence_map.jsonl")
    parser.add_argument("--gold-table", default="data/processed/gold_sentence_salience.csv")
    parser.add_argument("--rst-artifacts", default="data/artifacts/rst_artifacts.jsonl")
    parser.add_argument("--rst-image-manifest", default="data/artifacts/rst_image_manifest.csv")
    parser.add_argument("--rst-image-dir", default="data/artifacts/rst_images")
    parser.add_argument("--feature-table", default="data/processed/feature_table.csv")
    parser.add_argument("--rst-rs3-root", default=None, help="Output directory for RS3 files (RST trees)")
    parser.add_argument("--scored-feature-table", default="data/processed/feature_table_scored.csv")
    parser.add_argument("--feature-score-report", default="data/interim/feature_score_report.json")
    parser.add_argument("--transformer-dataset", default="data/processed/salience_transformer_dataset.csv")
    parser.add_argument("--transformer-split-report", default="data/interim/transformer_split_report.json")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-train", type=float, default=0.7)
    parser.add_argument("--split-val", type=float, default=0.15)
    parser.add_argument("--split-test", type=float, default=0.15)
    parser.add_argument("--psychformers-dir", default="")
    parser.add_argument("--psychformers-output-dir", default="data/interim/psychformers/output")
    parser.add_argument("--psychformers-model", default="gpt2")
    parser.add_argument("--psychformers-decoder", choices=["masked", "causal"], default="masked")
    parser.add_argument("--psychformers-following-context", action="store_true")
    parser.add_argument("--psychformers-use-cpu", action="store_true")
    parser.add_argument("--run-psychformers", action="store_true")
    parser.add_argument("--sample-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage = ALIASES.get(args.stage, args.stage)

    print(f"Running stage: {stage}")

    if stage == "sample_dataset":
        print(f"[DEBUG] Running sample_dataset stage with sample_size={args.sample_size}, output={args.sample_manifest}")
        sample_paragraphs(Path(args.sample_manifest), sample_size=args.sample_size, seed=args.seed)
        print("sample_dataset complete")
        return

    if stage == "clean_data":
        validate_and_clean(Path(args.sample_manifest), Path(args.cleaned_manifest), Path(args.cleaning_report))
        print("clean_data complete")
        return

    if stage == "segment_sentences":
        segment_and_align(Path(args.cleaned_manifest), Path(args.sentence_table), Path(args.segment_diagnostics))
        print("segment_sentences complete")
        return

    if stage == "build_gold_labels":
        map_answers(Path(args.cleaned_manifest), Path(args.sentence_table), Path(args.answer_mapping))
        build_gold_labels(Path(args.sentence_table), Path(args.answer_mapping), Path(args.gold_table))
        print("build_gold_labels complete")
        return

    if stage == "parse_rst_trees":
        rs3_root = Path(args.rst_rs3_root) if args.rst_rs3_root else None
        parse_rst_placeholder(Path(args.cleaned_manifest), Path(args.rst_artifacts), rs3_root=rs3_root)
        build_rst_image_manifest(Path(args.cleaned_manifest), Path(args.rst_image_dir), Path(args.rst_image_manifest))
        print("parse_rst_trees complete")
        return

    if stage == "feature_extraction":
        psychformers_dir = Path(args.psychformers_dir) if args.psychformers_dir else None
        extract_features(
            sentence_table_path=Path(args.sentence_table),
            gold_table_path=Path(args.gold_table),
            rst_artifacts_path=Path(args.rst_artifacts),
            output_csv_path=Path(args.feature_table),
            psychformers_dir=psychformers_dir,
            psychformers_output_dir=Path(args.psychformers_output_dir),
            psychformers_model=args.psychformers_model,
            psychformers_decoder=args.psychformers_decoder,
            psychformers_following_context=args.psychformers_following_context,
            psychformers_use_cpu=args.psychformers_use_cpu,
            run_psychformers=args.run_psychformers,
        )
        print("feature_extraction complete")
        return

    if stage == "score_feature_salience":
        score_feature_salience(
            feature_table_path=Path(args.feature_table),
            output_csv_path=Path(args.scored_feature_table),
            report_json_path=Path(args.feature_score_report),
        )
        print("score_feature_salience complete")
        return

    if stage == "prepare_transformer_dataset":
        prepare_transformer_dataset(
            source_feature_table_path=Path(args.feature_table),
            output_csv_path=Path(args.transformer_dataset),
            split_report_path=Path(args.transformer_split_report),
            split_seed=args.split_seed,
            split_train=args.split_train,
            split_val=args.split_val,
            split_test=args.split_test,
        )
        print("prepare_transformer_dataset complete")
        return


    if stage == "salience_classifier":
        # Import and run the salience classifier training
        from scripts.train_salience_classifier import main as train_salience_main
        train_salience_main()
        print("salience_classifier complete")
        return

    if stage == "llm_classifier":
        # Import and run the LLM classifier script (to be implemented/renamed)
        from scripts.llm_classifier import main as llm_classifier_main
        llm_classifier_main()
        print("llm_classifier complete")
        return

if __name__ == "__main__":
    main()

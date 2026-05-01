"""
run_inference_pipeline.py

Modular inference pipeline for processing new paragraphs through all stages:
- Sampling → Cleaning → Segmentation → Gold Label Generation → RST Parsing
- Feature Extraction → Dataset Preparation → Model Inference (Hybrid + LLM)
- Evaluation → Question Generation

All outputs are saved in data/inference for traceability.
"""

import subprocess
from pathlib import Path
import sys

def run_stage(stage, args_dict):
    """Run a pipeline stage using run_stage.py as a subprocess with custom arguments."""
    cmd = [sys.executable, "-m", "src.pipeline.run_stage", "--stage", stage]
    for k, v in args_dict.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.append(f"--{k}")
            cmd.append(str(v))
    print(f"[Pipeline] Running stage: {stage}")
    subprocess.run(cmd, check=True)

def run_inference_pipeline(sample_size=10, output_base_dir="data/inference"):
    seed = 42  # Default seed for reproducibility; can be parameterized
    output_dir = Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Paths for all outputs (no seed in filenames)
    sample_manifest_path = output_dir / "sample_manifest.json"
    cleaned_manifest_path = output_dir / "cleaned_sample_manifest.json"
    cleaning_report_path = output_dir / "cleaning_report.json"
    sentence_table_path = output_dir / "sentence_table.csv"
    segment_diagnostics_path = output_dir / "segmentation_diagnostics.json"
    mapping_jsonl_path = output_dir / "answer_sentence_map.jsonl"
    gold_output_csv = output_dir / "gold_sentence_salience.csv"
    rst_artifacts_path = output_dir / "rst_artifacts.jsonl"
    rst_image_manifest_path = output_dir / "rst_image_manifest.csv"
    rst_image_dir = output_dir / "rst_images"
    feature_table_csv = output_dir / "feature_table.csv"
    transformer_dataset_csv = output_dir / "salience_transformer_dataset.csv"
    transformer_split_report = output_dir / "transformer_split_report.json"
    # PsychFormers output directories for inference (hardcoded)
    psychformers_output_dir = output_dir / "psychformers" / "output"
    # RS3 output directory for RST parsing
    rst_rs3_root = output_dir / "rst_rs3"
    gold_path = output_dir / "gold_sentence_salience.csv"
    eval_hybrid_output = output_dir / "hybrid_eval_report.json"
    eval_llm_output = output_dir / "llm_eval_report.json"

    # --- PRE-PROCESSED STAGES (Commented out to resume salience stages) ---
    # 1. Sampling
    # run_stage("sample_dataset", {
    #     "sample-manifest": sample_manifest_path,
    #     "sample-size": sample_size
    # })

    # # 2. Cleaning
    # run_stage("clean_data", {
    #     "sample-manifest": sample_manifest_path,
    #     "cleaned-manifest": cleaned_manifest_path,
    #     "cleaning-report": cleaning_report_path
    # })

    # # 3. Sentence Segmentation
    # run_stage("segment_sentences", {
    #     "cleaned-manifest": cleaned_manifest_path,
    #     "sentence-table": sentence_table_path,
    #     "segment-diagnostics": segment_diagnostics_path
    # })

    # # 4. Gold Label Generation
    # run_stage("build_gold_labels", {
    #     "cleaned-manifest": cleaned_manifest_path,
    #     "sentence-table": sentence_table_path,
    #     "answer-mapping": mapping_jsonl_path,
    #     "gold-table": gold_output_csv
    # })

    # # 5. RST Parsing
    # run_stage("parse_rst_trees", {
    #     "cleaned-manifest": cleaned_manifest_path,
    #     "rst-artifacts": rst_artifacts_path,
    #     "rst-image-manifest": rst_image_manifest_path,
    #     "rst-image-dir": rst_image_dir,
    #     "rst-rs3-root": str(rst_rs3_root)
    # })

    # # 6. Feature Extraction
    # run_stage("feature_extraction", {
    #     "sentence-table": sentence_table_path,
    #     "gold-table": gold_output_csv,
    #     "rst-artifacts": rst_artifacts_path,
    #     "feature-table": feature_table_csv,
    #     "psychformers-output-dir": str(psychformers_output_dir),
    #     "run-psychformers": True,
    #     "psychformers-dir": "tools/PsychFormers",
    #     "psychformers-decoder": "causal"
    # })

    # # 7. Prepare Transformer Dataset
    # run_stage("prepare_transformer_dataset", {
    #     "feature-table": feature_table_csv,
    #     "transformer-dataset": transformer_dataset_csv,
    #     "transformer-split-report": transformer_split_report,
    #     "split-seed": seed,
    #     "split-train": 0.7,
    #     "split-val": 0.15,
    #     "split-test": 0.15
    # })
# 
    print("[Pipeline] Core pipeline stages complete. Running Hybrid and LLM inference...")

    # 8. Hybrid Inference
    hybrid_inference_output = output_dir / "feature_table_hybrid_inference.csv"
    print(f"[Pipeline] Running Hybrid RoBERTa inference on {transformer_dataset_csv}...")
    subprocess.run([
        sys.executable, "-m", "src.inference.hybrid_inference",
        str(transformer_dataset_csv), str(hybrid_inference_output)
    ], check=True)

    # 9. LLM Inference
    llm_inference_output = output_dir / "sentence_table_llm_inference.csv"
    print(f"[Pipeline] Running LLM inference on {sentence_table_path}...")
    subprocess.run([
        sys.executable, "-m", "src.inference.llm_inference",
        str(sentence_table_path), str(llm_inference_output)
    ], check=True)

    print("[Pipeline] Inference stages complete. Outputs saved:")
    print(f"  Hybrid: {hybrid_inference_output}")
    print(f"  LLM: {llm_inference_output}")

    # 10. Evaluation step: compare predictions to gold labels
    print("[Pipeline] Running evaluation against gold labels...")
    subprocess.run([
        sys.executable, "-m", "src.scoring.evaluate_inference",
        "--gold", str(gold_path),
        "--hybrid", str(hybrid_inference_output),
        "--llm", str(llm_inference_output),
        "--eval-hybrid", str(eval_hybrid_output),
        "--eval-llm", str(eval_llm_output)
    ], check=True)
    print(f"[Pipeline] Evaluation complete. Reports saved: {eval_hybrid_output}, {eval_llm_output}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the inference pipeline end-to-end.")
    parser.add_argument("--sample-size", type=int, default=15, help="Number of paragraphs to sample")
    args = parser.parse_args()
    run_inference_pipeline(sample_size=args.sample_size)
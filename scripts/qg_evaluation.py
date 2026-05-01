"""
Unified QG Evaluation Wrapper: Stage 1 (automatic) and Stage 2 (QA, LLM-as-a-Judge)
"""
import pandas as pd
import json
from pathlib import Path
from src.qg.evaluation import (
    load_gold_manifest,
    load_generated_questions,
    get_qa_pipeline,
    get_llm_judge,
    evaluate_stage1,
    evaluate_stage2
)

# ====================== CONFIG ======================
GOLD_FILE = Path("data/inference/cleaned_sample_manifest.json")
EXTRACTED_DIR = Path("data/qg/outputs")

MODES = {
    "zero_shot": EXTRACTED_DIR / "qg_zero_shot_extracted.json",
    "hybrid_salient": EXTRACTED_DIR / "qg_hybrid_salient_extracted.json",
    "llm_salient": EXTRACTED_DIR / "qg_llm_salient_extracted.json",
}
JUDGE_MODEL = "microsoft/Phi-4-mini-instruct"  # Use a Hugging Face free model (recommended)
# ===================================================

def main():
    gold_dict = load_gold_manifest(GOLD_FILE)
    results_stage1 = []
    results_stage2 = []
    all_judgments = {}
    qa_pipe = get_qa_pipeline()
    llm_judge = get_llm_judge(JUDGE_MODEL)

    for mode, file_path in MODES.items():
        print(f"\n[Stage 1] Evaluating {mode} (automatic metrics)...")
        gen_dict = load_generated_questions(file_path, gold_dict=gold_dict)
        auto_metrics, n = evaluate_stage1(gold_dict, gen_dict)
        results_stage1.append({"Mode": mode, **auto_metrics, "Num_Questions": n})

    for mode, file_path in MODES.items():
        print(f"\n[Stage 2] Evaluating {mode} (QA + LLM-as-a-Judge)...")
        gen_dict = load_generated_questions(file_path, gold_dict=gold_dict)
        summary, per_question = evaluate_stage2(
            gold_dict, gen_dict, qa_pipe, llm_judge, JUDGE_MODEL
        )
        summary["Mode"] = mode
        results_stage2.append(summary)
        all_judgments[mode] = per_question

    # Save Stage 1 summary
    df1 = pd.DataFrame(results_stage1)
    stage1_path = EXTRACTED_DIR / "qg_evaluation_stage1.csv"
    df1.to_csv(stage1_path, index=False)
    print("\n" + "="*85)
    print("STAGE 1 QG EVALUATION SUMMARY")
    print("="*85)
    print(df1.to_string(index=False))
    print(f"\nResults saved to: {stage1_path}")

    # Save Stage 2 summary
    df2 = pd.DataFrame(results_stage2)
    stage2_path = EXTRACTED_DIR / "qg_evaluation_stage2_summary.csv"
    df2.to_csv(stage2_path, index=False)
    print("\n" + "="*90)
    print("STAGE 2 QG EVALUATION SUMMARY")
    print("="*90)
    print(df2.to_string(index=False))
    print(f"\nResults saved to: {stage2_path}")

    # Save detailed judgments
    judgments_path = EXTRACTED_DIR / "qg_judgments_stage2.json"
    with open(judgments_path, "w", encoding="utf-8") as f:
        json.dump(all_judgments, f, indent=2, ensure_ascii=False)
    print(f"Per-question judgments saved to: {judgments_path}")

if __name__ == "__main__":
    main()

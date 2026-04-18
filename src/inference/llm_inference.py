"""
llm_inference.py

Module for running LLM-based salience inference on a sentence table CSV.
Uses the latest prompt and logic from llm_classifier.py (text-generation pipeline, robust output parsing).
"""
import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm
from pathlib import Path

def make_salience_prompt(passage: str, sentence: str) -> str:
    """Prompt for text-generation: instruct model to output SALIENT or NOT_SALIENT."""
    return (
        "You are an expert in identifying salient sentences for question-answering tasks. "
        "A sentence is salient if it contains the most important information that answers a potential question about the paragraph.\n\n"
        f"Given the following paragraph and one sentence from it, classify the sentence as salient or not salient.\n\n"
        f"<PARAGRAPH>\n{passage}\n</PARAGRAPH>\n\n<SENTENCE>\n{sentence}\n</SENTENCE>\n\n"
        "Answer with exactly one of these two options:\n- SALIENT \n- NOT_SALIENT \n\nClassification:"
    )

def extract_salience_label(generated: str) -> int | None:
    """Extracts 1 for SALIENT, 0 for NOT_SALIENT, or None if unclear from model output."""
    text = generated.lower()
    if "classification: salient" in text:
        return 1
    else:
        return 0

def run_llm_inference(sentence_table_csv: Path, output_csv: Path, model_name: str = "models/llm_model"):
    df = pd.read_csv(sentence_table_csv)
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-generation", model=model_name, device=device)
    para_groups = df.groupby("para_id")
    para_texts = {pid: " ".join(group["sent_text"].tolist()) for pid, group in para_groups}

    results = []
    generated_outputs = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="LLM Salience Classification"):
        sent = row["sent_text"]
        para_id = row["para_id"]
        passage = para_texts.get(para_id, "")
        prompt = make_salience_prompt(passage, sent)
        output = pipe(prompt, max_new_tokens=10)[0]
        generated = output["generated_text"].strip().upper()
        salience = extract_salience_label(generated)
        results.append(salience)
        generated_outputs.append({"para_id": para_id, "generated": generated})
    df["llm_salient"] = results
    # Save generated outputs to a separate JSON file
    import json
    gen_json_path = output_csv.with_suffix('.llm_generated.json')
    with open(gen_json_path, "w", encoding="utf-8") as f:
        json.dump(generated_outputs, f, indent=2, ensure_ascii=False)
    df.to_csv(output_csv, index=False)
    print(f"LLM-based salience tagging complete. Output saved to {output_csv}")
    print(f"Generated outputs saved to {gen_json_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LLM-based salience inference on a sentence table CSV.")
    parser.add_argument("sentence_table_csv", type=str, help="Path to the input sentence table CSV")
    parser.add_argument("output_csv", type=str, help="Path to save the output CSV with llm_salient column")
    parser.add_argument("--model-name", type=str, default="models/llm_model", help="Path or name of the LLM model")
    args = parser.parse_args()
    run_llm_inference(Path(args.sentence_table_csv), Path(args.output_csv), model_name=args.model_name)

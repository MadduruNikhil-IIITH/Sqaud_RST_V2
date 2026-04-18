

import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm
import os
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def make_salience_prompt(passage: str, sentence: str) -> str:
    """Prompt for text-generation: instruct model to output SALIENT or NOT_SALIENT."""
    return (
        "You are an expert in identifying salient sentences for question-answering tasks. "
        "A sentence is salient if it contains the most important information that answers a potential question about the paragraph.\n\n"
        f"Given the following paragraph and one sentence from it, classify the sentence as salient or not salient.\n\n"
        f"<PARAGRAPH>\n{passage}\n</PARAGRAPH>\n\n<SENTENCE>\n{sentence}\n</SENTENCE>\n\n"
        "Answer with exactly one of these two options:\n- SALIENT (salient)\n- NOT_SALIENT (not salient)\n\nClassification:"
    )

def extract_salience_label(generated: str) -> int | None:
    """Extracts 1 for SALIENT, 0 for NOT_SALIENT, or None if unclear from model output."""
    text = generated.lower()
    if "classification: salient" in text:
        return 1
    else:
        return 0

def main():
    df = pd.read_csv("data/processed/sentence_table.csv")
    # Use CUDA if available, else CPU

    device = 0 if torch.cuda.is_available() else -1
    model_save_dir = "models/llm_model"
    if os.path.exists(model_save_dir) and os.path.exists(os.path.join(model_save_dir, "config.json")):
        print(f"[INFO] Loading model and tokenizer from local directory: {model_save_dir}")
        pipe = pipeline("text-generation", model=model_save_dir, tokenizer=model_save_dir, device=device)
    else:
        model_name = "microsoft/Phi-4-mini-instruct"  # Efficient, not gated, fits 8GB VRAM
        print(f"[INFO] Downloading model from Hugging Face: {model_name}")
        pipe = pipeline("text-generation", model=model_name, device=device)

    # Save model and tokenizer for later inference
    model_save_dir = "models/llm_model"
    os.makedirs(model_save_dir, exist_ok=True)
    pipe.model.save_pretrained(model_save_dir)
    pipe.tokenizer.save_pretrained(model_save_dir)
    print(f"LLM model and tokenizer saved to {model_save_dir}")
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
    print("LLM classification complete. Adding results to DataFrame and saving...")
    df["llm_salient"] = results
    df.to_csv("data/processed/sentence_table_llm_scored.csv", index=False)
    import json
    gen_json_path = "data/processed/sentence_table_llm_generated.json"
    with open(gen_json_path, "w", encoding="utf-8") as f:
        json.dump(generated_outputs, f, indent=2, ensure_ascii=False)
    print(f"Generated outputs saved to {gen_json_path}")
    print("LLM-based salience tagging complete. Output saved to data/processed/sentence_table_llm_scored.csv")

    # --- Evaluation against gold labels ---
    gold_path = "data/processed/gold_sentence_salience.csv"
    if os.path.exists(gold_path):
        gold_df = pd.read_csv(gold_path)
        merged = pd.merge(df, gold_df[["sent_id", "gold_salient"]], on="sent_id", how="inner")
        y_true = merged["gold_salient"].fillna(0).astype(int)
        y_pred = merged["llm_salient"].fillna(0).astype(int)
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        eval_metrics = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_samples": int(len(y_true)),
        }
        os.makedirs("data/interim", exist_ok=True)
        with open("data/interim/llm_eval.json", "w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"LLM evaluation metrics saved to data/interim/llm_eval.json: {eval_metrics}")
    else:
        print("Gold labels not found; skipping evaluation.")

if __name__ == "__main__":
    main()
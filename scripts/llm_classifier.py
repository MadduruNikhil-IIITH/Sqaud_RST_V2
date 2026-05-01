"""
LLM Classifier for sentence salience (training/evaluation data).
V2: Uses paragraph-level ranking instead of per-sentence classification
to prevent over-selection.
"""
import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm
import os
import json
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def make_salience_prompt_ranked(passage: str, sentences: list[str]) -> str:
    """Paragraph-level ranking prompt: forces the LLM to compare all sentences
    and select only the most important 1-2 for QA salience."""
    sent_list = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    return (
        "<|system|>\n"
        "You are an expert at identifying the single most important sentence "
        "in a paragraph for SQuAD-style question answering. A salient sentence "
        "contains the specific factual detail that would most likely be the "
        "answer to a reading comprehension question.<|end|>\n"
        "<|user|>\n"
        "Given the paragraph below, select the 1 or 2 MOST salient sentences.\n"
        "Example:\n"
        "Paragraph: The Amazon rainforest is the largest in the world. It covers most of the Amazon basin. It was discovered by Orellana in 1541.\n"
        "Sentences:\n"
        "[0] The Amazon rainforest is the largest in the world.\n"
        "[1] It covers most of the Amazon basin.\n"
        "[2] It was discovered by Orellana in 1541.\n"
        "Salient sentence: [2]\n\n"
        "Paragraph:\n"
        f"{passage}\n\n"
        f"Sentences:\n{sent_list}\n\n"
        "Rules:\n"
        "- Select ONLY 1 or 2 sentences that contain the most specific, "
        "answerable factual information.\n"
        "- Do NOT select general/introductory sentences.\n"
        "- Do NOT select more than 2 sentences.\n"
        "- Return ONLY the sentence indices in square brackets.\n\n"
        "Example output: [2]\n"
        "Example output for two: [0, 3]\n\n"
        "Most salient sentence(s):<|end|>\n"
        "<|assistant|>"
    )


def extract_ranked_indices(generated_text: str, num_sentences: int) -> list[int]:
    """Parse the LLM output to extract selected sentence indices."""
    # Strategy 1: Find bracketed list like [2] or [0, 3]
    bracket_match = re.findall(r'\[([^\]]+)\]', generated_text)
    for match in bracket_match:
        try:
            indices = [int(x.strip()) for x in match.split(',')]
            valid = list(dict.fromkeys([i for i in indices if 0 <= i < num_sentences]))
            if valid:
                return valid[:2]
        except ValueError:
            continue

    # Strategy 2: Find standalone numbers
    numbers = re.findall(r'\b(\d+)\b', generated_text)
    valid = [int(n) for n in numbers if 0 <= int(n) < num_sentences]
    if valid:
        return list(dict.fromkeys(valid))[:2]

    # Strategy 3: Fallback — select sentence 0
    return [0]


def main():
    df = pd.read_csv("data/processed/sentence_table.csv")
    device = 0 if torch.cuda.is_available() else -1
    model_save_dir = "models/llm_model"

    if os.path.exists(model_save_dir) and os.path.exists(os.path.join(model_save_dir, "config.json")):
        print(f"[INFO] Loading model and tokenizer from local directory: {model_save_dir}")
        pipe = pipeline("text-generation", model=model_save_dir, tokenizer=model_save_dir, device=device)
    else:
        model_name = "microsoft/Phi-4-mini-instruct"
        print(f"[INFO] Downloading model from Hugging Face: {model_name}")
        pipe = pipeline("text-generation", model=model_name, device=device)

    # Save model and tokenizer for later inference
    os.makedirs(model_save_dir, exist_ok=True)
    pipe.model.save_pretrained(model_save_dir)
    pipe.tokenizer.save_pretrained(model_save_dir)
    print(f"LLM model and tokenizer saved to {model_save_dir}")

    # Group sentences by paragraph for ranked inference
    para_groups = df.groupby("para_id", sort=False)
    all_labels = [None] * len(df)
    generated_outputs = []

    for para_id, group in tqdm(para_groups, desc="LLM Salience Ranking"):
        sentences = group["sent_text"].tolist()
        passage = " ".join(sentences)

        prompt = make_salience_prompt_ranked(passage, sentences)

        try:
            output = pipe(prompt, max_new_tokens=30, do_sample=False, return_full_text=False)[0]
            generated_text = output["generated_text"].strip()
        except Exception as e:
            print(f"[ERROR] LLM inference failed for {para_id}: {e}")
            generated_text = "[0]"

        selected_indices = extract_ranked_indices(generated_text, len(sentences))

        for i, (idx, row) in enumerate(group.iterrows()):
            label = 1 if i in selected_indices else 0
            all_labels[idx] = label

        generated_outputs.append({
            "para_id": para_id,
            "num_sentences": len(sentences),
            "selected_indices": selected_indices,
            "generated": generated_text
        })

    df["llm_salient"] = all_labels
    df.to_csv("data/processed/sentence_table_llm_scored.csv", index=False)

    gen_json_path = "data/processed/sentence_table_llm_generated.json"
    with open(gen_json_path, "w", encoding="utf-8") as f:
        json.dump(generated_outputs, f, indent=2, ensure_ascii=False)

    pos_rate = df["llm_salient"].mean()
    print(f"LLM ranked salience tagging complete.")
    print(f"Positive rate: {pos_rate:.1%} ({df['llm_salient'].sum()}/{len(df)} sentences)")
    print(f"Output saved to data/processed/sentence_table_llm_scored.csv")
    print(f"Generated outputs saved to {gen_json_path}")

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
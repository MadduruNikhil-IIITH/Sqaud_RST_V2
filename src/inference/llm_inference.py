"""
llm_inference.py

Module for running LLM-based salience inference on a sentence table CSV.
V2: Uses paragraph-level ranking instead of per-sentence classification
to prevent over-selection (previous version marked 90% of sentences as salient).
"""
import re
import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm
from pathlib import Path


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
    """Parse the LLM output to extract selected sentence indices.
    Robust parsing with multiple fallback strategies."""
    # Strategy 1: Find bracketed list like [2] or [0, 3]
    bracket_match = re.findall(r'\[([^\]]+)\]', generated_text)
    for match in bracket_match:
        try:
            indices = [int(x.strip()) for x in match.split(',')]
            # Validate indices are in range
            valid = list(dict.fromkeys([i for i in indices if 0 <= i < num_sentences]))
            if valid:
                return valid[:2]  # Cap at 2
        except ValueError:
            continue

    # Strategy 2: Find standalone numbers
    numbers = re.findall(r'\b(\d+)\b', generated_text)
    valid = [int(n) for n in numbers if 0 <= int(n) < num_sentences]
    if valid:
        return list(dict.fromkeys(valid))[:2]  # Dedupe, cap at 2

    # Strategy 3: Fallback — select sentence 0 (first sentence, often topic sentence)
    return [0]


# Keep legacy per-sentence prompt for reference/fallback
def make_salience_prompt(passage: str, sentence: str) -> str:
    """Legacy prompt for text-generation: instruct model to output SALIENT or NOT_SALIENT."""
    return (
        "You are an expert in identifying salient sentences for question-answering tasks. "
        "A sentence is salient if it contains the most important information that answers a potential question about the paragraph.\n\n"
        f"Given the following paragraph and one sentence from it, classify the sentence as salient or not salient.\n\n"
        f"<PARAGRAPH>\n{passage}\n</PARAGRAPH>\n\n<SENTENCE>\n{sentence}\n</SENTENCE>\n\n"
        "Answer with exactly one of these two options:\n- SALIENT \n- NOT_SALIENT \n\nClassification:"
    )


def extract_salience_label(generated: str) -> int | None:
    """Extracts 1 for SALIENT, 0 for NOT_SALIENT from model output.
    Fixed: checks NOT_SALIENT before SALIENT to avoid substring match bug."""
    text = generated.lower()
    if "not_salient" in text or "not salient" in text:
        return 0
    elif "salient" in text:
        return 1
    else:
        return 0


def run_llm_inference(sentence_table_csv: Path, output_csv: Path, model_name: str = "models/llm_model"):
    df = pd.read_csv(sentence_table_csv)
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-generation", model=model_name, device=device)

    # Group sentences by paragraph for ranked inference
    para_groups = df.groupby("para_id", sort=False)

    all_labels = [None] * len(df)
    generated_outputs = []

    for para_id, group in tqdm(para_groups, desc="LLM Salience Ranking"):
        sentences = group["sent_text"].tolist()
        passage = " ".join(sentences)

        # Build paragraph-level ranking prompt
        prompt = make_salience_prompt_ranked(passage, sentences)

        try:
            output = pipe(prompt, max_new_tokens=30, do_sample=False, return_full_text=False)[0]
            generated_text = output["generated_text"].strip()
        except Exception as e:
            print(f"[ERROR] LLM inference failed for {para_id}: {e}")
            generated_text = "[0]"  # Fallback

        # Parse which sentence indices were selected
        selected_indices = extract_ranked_indices(generated_text, len(sentences))

        # Assign labels: 1 for selected, 0 for rest
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

    # Save generated outputs to a separate JSON file
    import json
    gen_json_path = output_csv.with_suffix('.llm_generated.json')
    with open(gen_json_path, "w", encoding="utf-8") as f:
        json.dump(generated_outputs, f, indent=2, ensure_ascii=False)

    df.to_csv(output_csv, index=False)

    # Print summary
    pos_rate = df["llm_salient"].mean()
    print(f"LLM ranked salience tagging complete. Output saved to {output_csv}")
    print(f"Generated outputs saved to {gen_json_path}")
    print(f"Positive rate: {pos_rate:.1%} ({df['llm_salient'].sum()}/{len(df)} sentences)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LLM-based salience inference on a sentence table CSV.")
    parser.add_argument("sentence_table_csv", type=str, help="Path to the input sentence table CSV")
    parser.add_argument("output_csv", type=str, help="Path to save the output CSV with llm_salient column")
    parser.add_argument("--model-name", type=str, default="models/llm_model", help="Path or name of the LLM model")
    args = parser.parse_args()
    run_llm_inference(Path(args.sentence_table_csv), Path(args.output_csv), model_name=args.model_name)

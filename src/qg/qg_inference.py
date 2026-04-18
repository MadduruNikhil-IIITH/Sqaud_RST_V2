"""
Optimized QG Inference for Phi-4-mini-instruct
Focus: Fair comparison of Hybrid Salience vs LLM Salience vs Zero-shot
"""

import os
import json
from tqdm import tqdm
import pandas as pd
from .model_loader import load_qg_model, BEST_MODEL


def construct_inputs(mode: str, manifest_df: pd.DataFrame, hybrid_df: pd.DataFrame, llm_df: pd.DataFrame):
    """Build list of input records for each mode."""
    # Ensure salience columns are integer
    if 'hybrid_salient' in hybrid_df.columns:
        hybrid_df['hybrid_salient'] = hybrid_df['hybrid_salient'].astype(int)
    if 'llm_salient' in llm_df.columns:
        llm_df['llm_salient'] = llm_df['llm_salient'].astype(int)

    inputs = []
    for _, row in manifest_df.iterrows():
        para_id = row['para_id']
        paragraph = row['context']

        if mode == "zero_shot":
            inputs.append({
                'para_id': para_id,
                'paragraph': paragraph,
                'mode': mode,
                'sentence': None
            })
        else:
            # Select salient sentences for hybrid or llm mode
            df = hybrid_df if mode == "hybrid_salient" else llm_df
            col = "hybrid_salient" if mode == "hybrid_salient" else "llm_salient"
            
            salient_rows = df[(df['para_id'] == para_id) & (df[col] == 1)]
            
            if len(salient_rows) == 0:
                # Fallback: use full paragraph if no salient sentence found
                inputs.append({
                    'para_id': para_id,
                    'paragraph': paragraph,
                    'mode': mode,
                    'sentence': paragraph[:500]  # truncate if too long
                })
            else:
                for _, srow in salient_rows.iterrows():
                    inputs.append({
                        'para_id': para_id,
                        'paragraph': paragraph,
                        'mode': mode,
                        'sentence': srow['sent_text']
                    })
    return inputs


def build_prompt(item: dict) -> str:
    """Build optimized chat-format prompt for Phi-4-mini-instruct."""
    paragraph = item['paragraph']
    mode = item['mode']
    sentence = item.get('sentence')

    system_msg = "You are an expert at generating high-quality, natural SQuAD-style questions."

    if mode == "zero_shot":
        user_msg = f"""Generate **one** clear, concise, and answerable question based only on the following paragraph.

Paragraph:
{paragraph}

Return your response as a single JSON object:
{{"question": "your generated question here"}}
"""
    else:
        user_msg = f"""Generate **one** clear, concise, and answerable question using the salient sentence and the full paragraph.

Salient sentence:
{sentence}

Full paragraph:
{paragraph}

Return your response as a single JSON object:
{{"question": "your generated question here", "sentence": "{sentence.replace('"', '\\"')}" }}
"""

    # Official Phi chat format
    prompt = f"""<|system|>
{system_msg}<|end|>
<|user|>
{user_msg}<|end|>
<|assistant|>"""

    return prompt


def run_qg_inference(
    model_name=BEST_MODEL,
    modes=None,
    input_dir="data/inference",
    output_dir="data/qg/outputs"
):
    if modes is None:
        modes = ["hybrid_salient", "llm_salient", "zero_shot"]

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Loading input files from: {input_dir}")

    with open(os.path.join(input_dir, "cleaned_sample_manifest.json"), "r", encoding="utf-8") as f:
        manifest_json = json.load(f)
    manifest_df = pd.DataFrame(manifest_json["paragraphs"])

    hybrid_df = pd.read_csv(os.path.join(input_dir, "feature_table_hybrid_inference.csv"))
    llm_df = pd.read_csv(os.path.join(input_dir, "sentence_table_llm_inference.csv"))

    print(f"[INFO] Loading model: {model_name}")
    pipe = load_qg_model(model_name)

    for mode in modes:
        print(f"[INFO] Processing mode: {mode}")
        inputs = construct_inputs(mode, manifest_df, hybrid_df, llm_df)
        print(f"[INFO] Number of prompts: {len(inputs)}")

        output_path = os.path.join(output_dir, f"qg_{mode}_generated.jsonl")

        with open(output_path, "w", encoding="utf-8") as fout:
            for idx, item in enumerate(tqdm(inputs, desc=f"QG {mode}")):
                prompt = build_prompt(item)

                if idx < 2:  # Show first 2 prompts for debugging
                    print(f"\n[DEBUG] Prompt for {mode} (idx {idx}):\n{prompt}\n---")

                try:
                    # Optimized generation params for Phi-4 (factual, low creativity)
                    outputs = pipe(
                        prompt,
                        max_new_tokens=120,
                        temperature=0.3,      # Low for consistency
                        do_sample=True,
                        top_p=0.9,
                        return_full_text=False
                    )

                    generated_text = outputs[0]['generated_text'].strip()

                    # Store raw output for debugging + try to extract question
                    item['generated_text'] = generated_text
                    item['question'] = extract_question(generated_text)

                except Exception as e:
                    print(f"[ERROR] Generation failed for item {idx}: {e}")
                    item['generated_text'] = f"[ERROR]: {e}"
                    item['question'] = ""

                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"[INFO] Finished mode '{mode}' → {output_path}")


def extract_question(text: str) -> str:
    """Simple extraction of the question from model output."""
    import re
    # Try to find JSON-like structure
    match = re.search(r'"question"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1).strip()
    
    # Fallback: take first line that looks like a question
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line and ('?' in line or line.lower().startswith('what') or 
                     line.lower().startswith('who') or line.lower().startswith('how')):
            return line
    return text[:200]  # fallback
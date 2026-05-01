"""
Updated Question Extraction Script
Compatible with the optimized QG inference using Phi-4-mini-instruct
"""

import json
import re
from pathlib import Path


def extract_question_from_output(generated_text: str) -> str:
    """Extract the actual question from model output with multiple fallback strategies."""
    if not generated_text or generated_text.strip() == "":
        return ""

    # 1. Try to parse JSON if model returned clean JSON
    try:
        parsed = json.loads(generated_text.strip())
        if isinstance(parsed, dict) and 'question' in parsed:
            return parsed['question'].strip()
        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            return parsed[0].get('question', '').strip()
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Regex for "question": "..." pattern
    match = re.search(r'"question"\s*:\s*"([^"]+)"', generated_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # 3. Take the first line that ends with a question mark
    lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
    for line in lines:
        if line.endswith('?') and len(line) > 10:
            return line

    # 4. Fallback: take first meaningful sentence
    sentences = re.split(r'[.!?]', generated_text)
    for sent in sentences:
        sent = sent.strip()
        if sent and len(sent) > 15:
            return sent + "?"

    # Final fallback: return first 150 characters
    return generated_text.strip()[:150]


def main():
    # Define input/output paths for all three modes
    configs = [
        {
            'mode': 'zero_shot',
            'input': 'data/qg/outputs/qg_zero_shot_generated.jsonl',
            'output': 'data/qg/outputs/qg_zero_shot_extracted.json'
        },
        {
            'mode': 'hybrid_salient',
            'input': 'data/qg/outputs/qg_hybrid_salient_generated.jsonl',
            'output': 'data/qg/outputs/qg_hybrid_salient_extracted.json'
        },
        {
            'mode': 'llm_salient',
            'input': 'data/qg/outputs/qg_llm_salient_generated.jsonl',
            'output': 'data/qg/outputs/qg_llm_salient_extracted.json'
        }
    ]

    for cfg in configs:
        mode = cfg['mode']
        input_path = Path(cfg['input'])
        output_path = Path(cfg['output'])

        if not input_path.exists():
            print(f"[WARNING] Input file not found: {input_path}")
            continue

        print(f"[INFO] Processing mode: {mode}")
        extracted_results = []

        with open(input_path, 'r', encoding='utf-8') as fin:
            for line_num, line in enumerate(fin, 1):
                try:
                    obj = json.loads(line.strip())
                    
                    para_id = obj.get('para_id')
                    paragraph = obj.get('paragraph')
                    sentence = obj.get('sentence')          # None for zero_shot
                    generated_text = obj.get('generated_text', obj.get('output', ''))
                    
                    question = extract_question_from_output(generated_text)

                    extracted_results.append({
                        'para_id': para_id,
                        'mode': mode,
                        'sentence': sentence,
                        'paragraph': paragraph,
                        'generated_question': question,
                        'generated_text': generated_text[:500] + "..." if len(generated_text) > 500 else generated_text
                    })

                except Exception as e:
                    print(f"[ERROR] Failed to process line {line_num} in {mode}: {e}")

        # Save extracted results
        with open(output_path, 'w', encoding='utf-8') as fout:
            json.dump(extracted_results, fout, ensure_ascii=False, indent=2)

        print(f"[INFO] Extracted {len(extracted_results)} questions for mode '{mode}' -> {output_path}")


if __name__ == "__main__":
    main()
    print("\n✅ Extraction completed for all three modes!")
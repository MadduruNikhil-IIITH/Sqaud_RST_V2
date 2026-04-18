"""
Modular QG Evaluation Core: Stage 1 (automatic) and Stage 2 (QA, LLM-as-a-Judge)
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from evaluate import load
from tqdm import tqdm
import torch
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForQuestionAnswering

# ====================== CONFIG ======================
QA_MODEL = "deepset/roberta-base-squad2"
# ===================================================

def load_gold_manifest(gold_file):
    with open(gold_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    gold = {}
    for para in data["paragraphs"]:
        para_id = para["para_id"]
        gold[para_id] = {
            "question": para["qas"][0]["question"],
            "answers": para["qas"][0]["answers"]["text"],
            "context": para["context"]
        }
    return gold

def load_generated_questions(file_path):
    if not Path(file_path).exists():
        print(f"[WARNING] File not found: {file_path}")
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=["para_id"], keep="first")
    return dict(zip(df["para_id"], df["generated_question"]))

def compute_automatic_metrics(references, predictions):
    rouge = load("rouge")
    bertscore = load("bertscore")
    meteor = load("meteor")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bert_scores = bertscore.compute(predictions=predictions, references=references, lang="en")
    meteor_scores = meteor.compute(predictions=predictions, references=references)
    return {
        "ROUGE-1": round(rouge_scores["rouge1"], 4),
        "ROUGE-2": round(rouge_scores["rouge2"], 4),
        "ROUGE-L": round(rouge_scores["rougeL"], 4),
        "BERTScore-F1": round(np.mean(bert_scores["f1"]), 4),
        "METEOR": round(meteor_scores["meteor"], 4),
    }

def get_qa_pipeline():
    """Load SQuAD QA model directly (deepset/roberta-base-squad2) for answerability evaluation."""
    model_name = "deepset/roberta-base-squad2"
    print(f"[INFO] Loading QA model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    print("[INFO] QA model loaded successfully")
    return tokenizer, model

def compute_qa_metrics(qa_tuple, context, question, gold_answers):
    tokenizer, model = qa_tuple
    try:
        inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        answer_tokens = inputs.input_ids[0, start_idx:end_idx+1]
        pred_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        em = int(any(pred_answer.lower() == ans.lower() for ans in gold_answers))
        def f1_score(pred, gold):
            pred_toks = pred.lower().split()
            gold_toks = gold.lower().split()
            common = set(pred_toks) & set(gold_toks)
            if not pred_toks or not gold_toks:
                return 0.0
            prec = len(common) / len(pred_toks)
            rec = len(common) / len(gold_toks)
            if prec + rec == 0:
                return 0.0
            return 2 * prec * rec / (prec + rec)
        f1 = max(f1_score(pred_answer, gold) for gold in gold_answers)
        return em, f1, pred_answer
    except Exception as e:
        print(f"[QA ERROR] {e}")
        return 0, 0.0, ""
    
def get_device():
    return 0 if torch.cuda.is_available() else -1

def get_llm_judge(judge_model):
    device = get_device()
    pipe_type = "text-generation"
    print(f"[INFO] Using pipeline type: {pipe_type}")

    return pipeline(
        pipe_type,
        model=judge_model,
        tokenizer=judge_model,
        device=device
    )

def build_judge_prompt(paragraph, generated_question, gold_question):
     prompt = f"""
You are an expert SQuAD-style question evaluator. Carefully judge the following generated question for answerability and quality. Use step-by-step reasoning and the rubrics below. Output ONLY a valid JSON object as shown.

---
Paragraph:
{paragraph}

Generated Question:
{generated_question}

Gold Question (for reference):
{gold_question}

---
Scoring Rubrics (1-5, with anchors):

1. Answerability (Primary):
    1 = Cannot be answered from paragraph, or answer is missing/ambiguous
    3 = Partially answerable, but unclear or incomplete
    5 = Clearly and fully answerable from paragraph

2. Reasonableness / Validity:
    1 = Illogical, spammy, or nonsensical
    3 = Somewhat reasonable, but awkward or odd
    5 = Fully logical, valid, and natural

3. Clarity & Naturalness:
    1 = Unclear, confusing, or unnatural
    3 = Somewhat clear, but awkward
    5 = Very clear, concise, and natural

4. Difficulty:
    1 = Trivial or impossible
    3 = Too easy or too hard, but not extreme
    5 = Challenging but fair for the context

5. Overall Quality:
    1 = Poor overall
    3 = Mixed/average
    5 = Excellent SQuAD-style question

Instructions:
- Think step by step (Chain-of-Thought).
- Output ONLY a valid JSON object with this structure, and place it between <JSON> and </JSON> tags:
<JSON>
{{
  "reasoning": "step-by-step reasoning",
  "scores": {{
     "answerability": X,
     "reasonableness": X,
     "clarity": X,
     "difficulty": X
  }},
  "overall_quality": X,
  "summary_comment": "one-sentence summary"
}}
</JSON>
"""
     return prompt

def call_llm_judge(llm, prompt):
    try:
        output = llm(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
        print("\n[LLM RAW OUTPUT]\n" + output + "\n[END RAW OUTPUT]\n")
        # Extract all <JSON>...</JSON> blocks and use the last one
        import re
        json_tag_blocks = re.findall(r'<JSON>\s*(\{.*?\})\s*</JSON>', output, re.DOTALL)
        if json_tag_blocks:
            json_str = json_tag_blocks[-1]
            try:
                return json.loads(json_str)
            except Exception as e2:
                print(f"[LLM JSON PARSE ERROR] {e2}\nExtracted: {json_str}")
                return None
        else:
            print("[LLM ERROR] No <JSON>...</JSON> block found in output.")
            return None
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return None

def evaluate_stage1(gold_dict, gen_dict):
    # Only automatic metrics
    common_ids = list(set(gold_dict.keys()) & set(gen_dict.keys()))
    references = [gold_dict[pid]["question"] for pid in common_ids]
    predictions = [gen_dict[pid] for pid in common_ids]
    auto_metrics = compute_automatic_metrics(references, predictions)
    return auto_metrics, len(common_ids)

def evaluate_stage2(gold_dict, gen_dict, qa_pipe, llm_judge, judge_model):
    common_ids = list(set(gold_dict.keys()) & set(gen_dict.keys()))
    references = [gold_dict[pid]["question"] for pid in common_ids]
    predictions = [gen_dict[pid] for pid in common_ids]
    auto_metrics = compute_automatic_metrics(references, predictions)
    ems, f1s = [], []
    for pid in tqdm(common_ids, desc="QA"):
        context = gold_dict[pid]["context"]
        question = gen_dict[pid]
        gold_answers = gold_dict[pid]["answers"]
        em, f1, _ = compute_qa_metrics(qa_pipe, context, question, gold_answers)
        ems.append(em)
        f1s.append(f1)
    qa_metrics = {
        "QA_EM": round(np.mean(ems), 4),
        "QA_F1": round(np.mean(f1s), 4)
    }
    judge_scores = {k: [] for k in ["answerability", "reasonableness", "clarity", "difficulty", "overall_quality"]}
    per_question = []
    for pid in tqdm(common_ids, desc="LLM Judge"):
        context = gold_dict[pid]["context"]
        gold_q = gold_dict[pid]["question"]
        gen_q = gen_dict[pid]
        prompt = build_judge_prompt(context, gen_q, gold_q)
        judgment = call_llm_judge(llm_judge, prompt)
        if judgment is None:
            continue
        scores = judgment.get("scores", {})
        for k in judge_scores:
            if k in scores:
                judge_scores[k].append(scores[k])
        per_question.append({
            "para_id": pid,
            "generated_question": gen_q,
            "gold_question": gold_q,
            "scores": scores,
            "overall_quality": judgment.get("overall_quality"),
            "reasoning": judgment.get("reasoning"),
            "summary_comment": judgment.get("summary_comment")
        })
    judge_means = {f"LLM_{k}_mean": round(np.mean(judge_scores[k]), 3) if judge_scores[k] else None for k in judge_scores}
    judge_stds = {f"LLM_{k}_std": round(np.std(judge_scores[k]), 3) if judge_scores[k] else None for k in judge_scores}
    summary = {
        **auto_metrics,
        **qa_metrics,
        **judge_means,
        **judge_stds,
        "Num_Questions": len(common_ids)
    }
    return summary, per_question

"""Quick script to analyze actual LLM responses vs parsed labels."""
import json
from collections import Counter

data = json.load(open("data/inference/sentence_table_llm_inference.llm_generated.json", encoding="utf-8"))

endings = []
for d in data:
    text = d["generated"]
    # Get the final classification token
    if "CLASSIFICATION: " in text:
        ending = text.split("CLASSIFICATION: ")[-1].strip()
    else:
        ending = "UNKNOWN"
    endings.append(ending)

print("=== LLM Actual Model Responses ===")
print(Counter(endings))
print(f"\nTotal sentences: {len(data)}")
print(f"Model said SALIENT: {sum(1 for e in endings if e == 'SALIENT')}")
print(f"Model said NOT_SALIENT: {sum(1 for e in endings if e == 'NOT_SALIENT')}")

# Now test the BUGGY extract function
def buggy_extract(generated):
    text = generated.lower()
    if "classification: salient" in text:
        return 1
    else:
        return 0

# And the FIXED extract function
def fixed_extract(generated):
    text = generated.lower()
    if "classification: not_salient" in text:
        return 0
    elif "classification: salient" in text:
        return 1
    else:
        return 0

print("\n=== Bug Analysis ===")
buggy = [buggy_extract(d["generated"]) for d in data]
fixed = [fixed_extract(d["generated"]) for d in data]

print(f"Buggy parser: {sum(buggy)} salient / {len(buggy)} total ({100*sum(buggy)/len(buggy):.0f}%)")
print(f"Fixed parser: {sum(fixed)} salient / {len(fixed)} total ({100*sum(fixed)/len(fixed):.0f}%)")
print(f"\nDifference: {sum(buggy) - sum(fixed)} sentences flipped from 1->0")

# Show which ones differ
print("\n=== Sentences that would change ===")
import pandas as pd
df = pd.read_csv("data/inference/sentence_table_llm_inference.csv")
for i, (b, f) in enumerate(zip(buggy, fixed)):
    if b != f:
        print(f"  [{i}] {data[i]['para_id']} | model said: {endings[i]} | buggy={b} fixed={f}")
        print(f"       Sentence: {df.iloc[i]['sent_text'][:80]}...")

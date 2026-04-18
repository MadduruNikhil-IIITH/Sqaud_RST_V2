# QG Inference Prompts by Mode

This document describes the exact prompts used for Question Generation (QG) inference in each mode. These prompts are optimized for use with chat-style models such as Phi-4-mini-instruct and ensure fair, consistent comparison across modes.

---

## Zero-Shot Mode

**Prompt Format:**

```
<|system|>
You are an expert at generating high-quality, natural SQuAD-style questions.<|end|>
<|user|>
Generate **one** clear, concise, and answerable question based only on the following paragraph.

Paragraph:
{paragraph}

Return your response as a single JSON object:
{"question": "your generated question here"}
<|end|>
<|assistant|>
```

- **{paragraph}**: The full paragraph from which to generate the question.
- **Output**: JSON object with a single key `question`.

---

## Hybrid Salient / LLM Salient Modes

**Prompt Format:**

```
<|system|>
You are an expert at generating high-quality, natural SQuAD-style questions.<|end|>
<|user|>
Generate **one** clear, concise, and answerable question using the salient sentence and the full paragraph.

Salient sentence:
{sentence}

Full paragraph:
{paragraph}

Return your response as a single JSON object:
{"question": "your generated question here", "sentence": "{sentence}" }
<|end|>
<|assistant|>
```

- **{sentence}**: The salient sentence (from hybrid or LLM salience method).
- **{paragraph}**: The full paragraph.
- **Output**: JSON object with keys `question` and `sentence`.

---

## Notes
- All prompts use the official Phi chat format for best compatibility.
- The model is instructed to return only the required JSON object for easy parsing.
- For fallback (if no salient sentence is found), the full paragraph is used as the `sentence` field (truncated to 500 characters).

---

For further details, see the implementation in `src/qg/qg_inference.py`.

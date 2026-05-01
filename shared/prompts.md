# Model Prompts for Salience & Question Generation

This document outlines the specific prompts and instructions used for the LLM-as-a-Judge salience classifier and the Phi-4 Question Generation pipeline.

---

## 1. Salience Ranking Prompt (LLM-as-a-Judge)
**Model**: `microsoft/Phi-4-mini-instruct`  
**Purpose**: Paragraph-level ranking to identify the top-2 salient sentences.

### System Message:
```text
You are an expert at identifying the single most important sentence in a paragraph for SQuAD-style question answering. A salient sentence contains the specific factual detail that would most likely be the answer to a reading comprehension question.
```

### User Prompt Template:
```text
Given the paragraph below, select the 1 or 2 MOST salient sentences.
Most paragraphs have only 1 truly salient sentence.

Paragraph:
{passage_text}

Sentences:
[0] Sentence 1 text...
[1] Sentence 2 text...
...

Rules:
- Select ONLY 1 or 2 sentences that contain the most specific, answerable factual information.
- Do NOT select general/introductory sentences.
- Do NOT select more than 2 sentences.
- Return ONLY the sentence indices in square brackets.

Example output: [2]
Example output for two: [0, 3]

Most salient sentence(s):
```

---

## 2. Question Generation Prompt
**Model**: `microsoft/Phi-4-mini-instruct`  
**Purpose**: Generating a SQuAD-style question targeted at a specific salient sentence.

### System Message:
```text
You are an expert at generating high-quality SQuAD-style questions.
Given a paragraph and one specific salient sentence from it, your task is to generate a question that is directly answerable from that sentence.
```

### User Prompt Template:
```text
Generate a high-quality SQuAD-style question for the given salient sentence within its full paragraph context.

Guidelines:
1. The question must be directly answerable from the information in the 'Salient Sentence'.
2. The question should be clear, natural-sounding, and objective.
3. Use the 'Full Paragraph' only for context; do not ask about information outside the salient sentence unless necessary for clarity.

###
Salient Sentence: "{salient_sentence}"

Full Paragraph:
{paragraph_text}

Output Format:
{"question": "your generated question here", "sentence": "{salient_sentence}" }
```

---

## 3. LLM-as-a-Judge Evaluation Prompt
**Model**: `microsoft/Phi-4-mini-instruct`  
**Purpose**: Scoring generated questions on a scale of 1-5 across multiple dimensions.

### Prompt Template:
```text
You are an expert evaluator of reading comprehension questions.
Rate the following generated question based on the provided paragraph.

Paragraph: {paragraph}
Question: {question}

Criteria (1-5 scale):
1. Answerability: Can the question be fully answered using ONLY the paragraph?
2. Reasonableness: Is the question logical and valid?
3. Clarity & Naturalness: Is the question well-phrased and natural?
4. Difficulty: 1=Trivial, 3=Fair, 5=Challenging.
5. Overall Quality: SQuAD-style excellence.

Output ONLY a valid JSON object:
<JSON>
{
  "reasoning": "...",
  "scores": { "answerability": X, "reasonableness": X, "clarity": X, "difficulty": X },
  "overall_quality": X,
  "summary_comment": "..."
}
</JSON>
```

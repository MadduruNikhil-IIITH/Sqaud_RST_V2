# QG Evaluation Metrics Reference

This document describes the evaluation metrics and rubrics used in the SQuAD Question Generation (QG) evaluation pipeline.

## Automatic Metrics

- **ROUGE (ROUGE-1, ROUGE-2, ROUGE-L):**
  - Measures n-gram overlap between generated and reference questions.
  - ROUGE-1: Unigram overlap
  - ROUGE-2: Bigram overlap
  - ROUGE-L: Longest common subsequence

- **BERTScore:**
  - Uses contextual embeddings to compute similarity between generated and reference questions.
  - Reports F1 score as the main metric.

- **METEOR:**
  - Considers exact, stem, synonym, and paraphrase matches between generated and reference questions.

## LLM-as-a-Judge Metrics (Rubric-based, 1-5 scale)

Each generated question is evaluated by an LLM using the following criteria:

1. **Salience Focus**
   - 1: Not about salient sentence at all
   - 3: Partially about salient sentence
   - 5: Directly and primarily about salient sentence

2. **Answerability**
   - 1: Cannot be answered from paragraph
   - 3: Answer is ambiguous or incomplete
   - 5: Clearly and fully answerable from paragraph

3. **Reasonableness/Validity**
   - 1: Illogical, spammy, or nonsensical
   - 3: Somewhat reasonable, but awkward or odd
   - 5: Fully logical, valid, and natural

4. **Difficulty**
   - 1: Trivial or impossible
   - 3: Too easy or too hard, but not extreme
   - 5: Challenging but fair for the context

5. **Clarity**
   - 1: Unclear, confusing, or unnatural
   - 3: Somewhat clear, but awkward
   - 5: Very clear, concise, and natural

### Output Structure

The LLM outputs a JSON object for each question with:
- `reasoning`: Step-by-step reasoning
- `scores`: Dictionary of rubric scores
- `overall_quality`: Overall score (1-5)
- `summary_comment`: One-sentence summary

---

For more details, see the evaluation script and code in `src/qg_evaluation/`.
# Future Directions: SQuAD-RST Question Generation

Based on our current results (Stage 2 Evaluation), here are the recommended next steps for the research group.

## 1. Large-Scale Evaluation
The current results are based on a 10-sample test (6 retained paragraphs). To ensure the findings are robust:
- **Task**: Run the full pipeline on a random sample of 100-200 paragraphs.
- **Goal**: Confirm if the METEOR and LLM Clarity gains for the `llm_salient` mode hold at scale.

## 2. Salience Alignment (Few-Shot Tuning)
We observed a 0% recall for LLM Salience against SQuAD Gold labels. 
- **Hypothesis**: The LLM picks "thematically salient" sentences (topic sentences), while SQuAD needs "factually salient" sentences (details).
- **Task**: Update the `llm_inference.py` prompt with 2-3 "SQuAD-style" examples where a detail sentence is chosen over a topic sentence.

## 3. RST-Weighted Generation
Currently, we use a binary selection (Salient or Not).
- **Task**: Use the `span_importance_score` (from the feature table) as a continuous weight for the generation probability.
- **Goal**: Test if questions generated from "Deep Nuclei" are more "answerable" than those from "Satellite" branches.

## 4. Cross-Dataset Generalization
- **Task**: Run the Hybrid Salient model on **HotpotQA** or **StrategyQA** without retraining.
- **Goal**: Test if discourse-based salience is a universal feature for reasoning.

## 5. Automated Pipeline Optimization
- **Task**: Refine the `run_inference_pipeline.py` to handle GPU memory more efficiently (e.g., clearing cache between modes).
- **Goal**: Allow for 1000+ sample runs on a single consumer GPU.

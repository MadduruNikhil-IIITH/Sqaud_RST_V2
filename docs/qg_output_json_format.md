# QG Output JSON Format Documentation

## General Principles
- All outputs should be structured JSON for robust downstream use.
- Each passage (paragraph) is represented as a single JSON object, grouping all questions for that passage.
- Each question entry includes the question text, and for salient modes, the salient sentence.

---

## 1. Zero-Shot Mode
### Structure
- One object per paragraph.
- Properties:
  - `para_id`: string (unique paragraph ID)
  - `paragraph`: string (the full paragraph text)
  - `questions`: array of strings (each is a generated question)

### Example
```json
{
  "para_id": "squad_v2_5727ebe03acd2414000deff0",
  "paragraph": "Nasser remains an iconic figure...",
  "questions": [
    "What were Nasser's contributions to social justice?",
    "What major projects did Nasser launch?",
    "How do historians view Nasser's legacy?"
  ]
}
```

---

## 2. Salient Modes (hybrid_salient, llm_salient)
### Structure
- One object per paragraph.
- Properties:
  - `para_id`: string (unique paragraph ID)
  - `paragraph`: string (the full paragraph text)
  - `qa_pairs`: array of objects, each with:
    - `question`: string (the generated question)
    - `salient_sentence`: string (the sentence used for question generation)

### Example
```json
{
  "para_id": "squad_v2_5727ebe03acd2414000deff0",
  "paragraph": "Nasser remains an iconic figure...",
  "qa_pairs": [
    {
      "question": "What are the reasons Nasser is considered an iconic figure in the Arab world?",
      "salient_sentence": "Nasser remains an iconic figure in the Arab world, particularly for his strides towards social justice and Arab unity, modernization policies, and anti-imperialist efforts."
    },
    {
      "question": "What are some criticisms leveled against Nasser's legacy according to his detractors?",
      "salient_sentence": "Nasser's detractors criticize his authoritarianism, his government's human rights violations, his populist relationship with the citizenry, and his failure to establish civil institutions, blaming his legacy for future dictatorial governance in Egypt."
    }
  ]
}
```

---

## Notes
- All fields are required for each entry.
- For zero_shot, `questions` is a list of strings.
- For salient modes, `qa_pairs` is a list of objects, each with both `question` and `salient_sentence`.
- This format enables easy grouping, evaluation, and downstream use.

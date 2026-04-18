import pandas as pd
ft = pd.read_csv("data/processed/feature_table.csv")

# Check for answer columns if present
answer_cols = ["answer_text", "answer_start_char", "answer_end_char"]
present_answer_cols = [col for col in answer_cols if col in ft.columns]
if present_answer_cols:
    summary = {}
    for col in present_answer_cols:
        empty = ft[col].isnull().sum() + (ft[col] == "").sum()
        non_empty = len(ft) - empty
        summary[col] = {"total": len(ft), "empty": int(empty), "non_empty": int(non_empty)}
    print("Answer column summary:", summary)
    print(ft[ft[present_answer_cols[0]].notnull() & (ft[present_answer_cols[0]] != "")][present_answer_cols].head())
else:
    print("No answer columns present.")


# Columns that are always empty
empty_cols = [col for col in ft.columns if ft[col].isnull().all() or (ft[col] == '').all()]
print('Columns always empty:', empty_cols)


# Columns likely not available at inference
suspect_cols = [col for col in ft.columns if any(x in col for x in ['answer', 'gold', 'rank', 'label', 'split', 'question_id'])]
print('Columns likely not available at inference:', suspect_cols)



# Show a summary of all columns with % non-empty
for col in ft.columns:
    non_empty = ft[col].notnull().sum() - (ft[col] == '').sum()
    print(f'{col}: {non_empty/len(ft)*100:.1f}% non-empty')


# Check for columns with only one unique value (constant columns)
constant_cols = [col for col in ft.columns if ft[col].nunique(dropna=False) == 1]
if constant_cols:
    print("Columns with only one unique value (no diversity):")
    for col in constant_cols:
        print(f"  - {col}: {ft[col].iloc[0]}")
else:
    print("All columns have diversity (more than one unique value).")

# Print all columns in the feature table
print("\nAll columns in feature_table.csv:")
for col in ft.columns:
    print(f"  - {col}")

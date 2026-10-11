import pandas as pd
from pathlib import Path

dataset_path = Path("data/job_resume_fit.csv")

print("CareerLens AI Dataset Inspection Started!")

if not dataset_path.exists():
    print("Dataset not found:", dataset_path.resolve())
    raise SystemExit(1)

df = pd.read_csv(dataset_path)

print("\n========== DATASET OVERVIEW ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== JOB CATEGORIES ==========")
print(df["category"].value_counts())

print("\n========== SAMPLE RECORDS ==========")
print(
    df[
        [
            "category",
            "ai_match_score",
            "skill_string_match_score",
            "fuzzy_match_score",
        ]
    ].head(5).to_string(index=False)
)

print("\nDataset inspection completed!")
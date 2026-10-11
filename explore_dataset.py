import pandas as pd
from pathlib import Path

# Load the dataset
path = Path("data/job_resume_fit.csv")
df = pd.read_csv(path)

print("\n========== SCORE SUMMARY ==========")

score_columns = [
    "ai_match_score",
    "skill_string_match_score",
    "fuzzy_match_score"
]

print(df[score_columns].describe().round(2))

print("\n========== SCORE CORRELATIONS ==========")
print(df[score_columns].corr().round(3))

print("\n========== CATEGORY EXAMPLES ==========")

for category in [
    "INFORMATION-TECHNOLOGY",
    "FINANCE",
    "HEALTHCARE"
]:
    sample = df[df["category"] == category].head(1)

    if not sample.empty:
        row = sample.iloc[0]

        print(f"\nCategory: {category}")
        print("Resume preview:")
        print(str(row["resume_text"])[:300])

        print("\nJob description preview:")
        print(str(row["job_text"])[:300])

        print("\nRequired skills:")
        print(str(row["job_required_skills"])[:300])

        print("\nAI match score:", row["ai_match_score"])

print("\n========== EXPLORATION COMPLETE ==========")

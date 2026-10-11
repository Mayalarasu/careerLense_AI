import pandas as pd

from skill_matcher import find_skills

df = pd.read_csv("data/job_resume_fit.csv")

print("Checking skill dictionary coverage...", flush=True)
print("Dataset records:", len(df), flush=True)

resume_counts = []
job_counts = []

for index, row in enumerate(
    df[["resume_text", "job_text"]].itertuples(index=False, name=None),
    start=1,
):
    resume_text, job_text = row

    resume_counts.append(
        len(find_skills(str(resume_text)))
    )

    job_counts.append(
        len(find_skills(str(job_text)))
    )

    if index % 100 == 0:
        print(f"Processed {index}/{len(df)} records", flush=True)

df["recognized_resume_skills"] = resume_counts
df["recognized_job_skills"] = job_counts

print("\n========== OVERALL COVERAGE ==========")

print(
    "Average recognized resume skills:",
    round(df["recognized_resume_skills"].mean(), 2),
)

print(
    "Average recognized job skills:",
    round(df["recognized_job_skills"].mean(), 2),
)

print(
    "Jobs with zero recognized skills:",
    int((df["recognized_job_skills"] == 0).sum()),
)

print("\n========== COVERAGE BY CATEGORY ==========")

summary = (
    df.groupby("category")
    .agg(
        records=("category", "size"),
        avg_resume_skills=("recognized_resume_skills", "mean"),
        avg_job_skills=("recognized_job_skills", "mean"),
        jobs_with_zero_skills=(
            "recognized_job_skills",
            lambda s: (s == 0).sum(),
        ),
    )
    .round(2)
    .sort_values("avg_job_skills")
)

print(summary.to_string())

summary.to_csv(
    "reports/skill_coverage_by_category.csv",
    index=True,
)

print("\nSaved reports/skill_coverage_by_category.csv")
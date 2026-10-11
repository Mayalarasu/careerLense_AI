import pandas as pd

from skill_matcher import find_skills

DATA_PATH = "data/job_resume_fit.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

resume_skill_sets = []
job_skill_sets = []

for index, row in enumerate(
    df[["resume_text", "job_text"]].itertuples(
        index=False,
        name=None,
    ),
    start=1,
):
    resume_text, job_text = row

    resume_skills = set(find_skills(str(resume_text)))
    job_skills = set(find_skills(str(job_text)))

    resume_skill_sets.append(resume_skills)
    job_skill_sets.append(job_skills)

    if index % 100 == 0:
        print(f"Processed {index}/{len(df)} records")

df["resume_skills_found"] = resume_skill_sets
df["job_skills_found"] = job_skill_sets

df["resume_skill_count"] = df["resume_skills_found"].apply(len)
df["job_skill_count"] = df["job_skills_found"].apply(len)

df["overlap_count"] = df.apply(
    lambda row: len(
        row["resume_skills_found"]
        & row["job_skills_found"]
    ),
    axis=1,
)

df["skill_overlap_pct"] = df.apply(
    lambda row: (
        row["overlap_count"] / row["job_skill_count"] * 100
        if row["job_skill_count"] > 0
        else 0
    ),
    axis=1,
)

print("\n========== OVERALL SKILL OVERLAP ==========")

print(
    "Average resume skills:",
    round(df["resume_skill_count"].mean(), 2),
)

print(
    "Average job skills:",
    round(df["job_skill_count"].mean(), 2),
)

print(
    "Average shared skills:",
    round(df["overlap_count"].mean(), 2),
)

print(
    "Average recognized job-skill overlap:",
    round(df["skill_overlap_pct"].mean(), 2),
    "%",
)

print("\n========== EXAMPLE RECORDS ==========")

for _, row in df.head(5).iterrows():
    print("\nCategory:", row["category"])
    print("Resume skills:", sorted(row["resume_skills_found"]))
    print("Job skills:", sorted(row["job_skills_found"]))
    print(
        "Shared skills:",
        sorted(
            row["resume_skills_found"]
            & row["job_skills_found"]
        ),
    )
    print(
        "Recognized job-skill overlap:",
        round(row["skill_overlap_pct"], 2),
        "%",
    )

print("\n========== OVERLAP BY CATEGORY ==========")

summary = (
    df.groupby("category")
    .agg(
        records=("category", "size"),
        avg_resume_skills=("resume_skill_count", "mean"),
        avg_job_skills=("job_skill_count", "mean"),
        avg_shared_skills=("overlap_count", "mean"),
        avg_overlap_pct=("skill_overlap_pct", "mean"),
    )
    .round(2)
)

print(summary.to_string())

summary.to_csv("reports/skill_overlap_by_category.csv")

print("\nSaved reports/skill_overlap_by_category.csv")
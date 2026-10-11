from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer


# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

DATA_PATH = Path("data/job_resume_fit.csv")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["resume_text", "job_text", "ai_match_score"]
).copy()

df["resume_text"] = df["resume_text"].astype(str)
df["job_text"] = df["job_text"].astype(str)
df["ai_match_score"] = pd.to_numeric(
    df["ai_match_score"],
    errors="coerce",
)

df = df.dropna(subset=["ai_match_score"])
df = df[df["ai_match_score"].between(0, 100)].copy()

print("Usable records:", len(df))


# --------------------------------------------------
# 2. Create text and explicit skill-overlap features
# --------------------------------------------------

from skill_matcher import find_skills

print("Extracting skills...")

resume_skills_list = []
job_skills_list = []
overlap_features = []

for index, row in enumerate(
    df[["resume_text", "job_text"]].itertuples(
        index=False,
        name=None,
    ),
    start=1,
):
    resume_text, job_text = row

    resume_skills = set(find_skills(resume_text))
    job_skills = set(find_skills(job_text))

    shared_skills = resume_skills & job_skills

    resume_skills_list.append(resume_skills)
    job_skills_list.append(job_skills)

    overlap_features.append({
        "resume_skill_count": len(resume_skills),
        "job_skill_count": len(job_skills),
        "shared_skill_count": len(shared_skills),
        "job_skill_overlap": (
            len(shared_skills) / len(job_skills)
            if job_skills else 0.0
        ),
        "resume_skill_coverage": (
            len(shared_skills) / len(resume_skills)
            if resume_skills else 0.0
        ),
    })

    if index % 100 == 0:
        print(f"Processed {index}/{len(df)} records")

df["combined_text"] = (
    "RESUME: "
    + df["resume_text"]
    + " JOB DESCRIPTION: "
    + df["job_text"]
)

feature_df = pd.DataFrame(
    overlap_features,
    index=df.index,
)

X_text = df["combined_text"]
y = df["ai_match_score"]


# --------------------------------------------------
# 3. Use exactly the same split as Model V1
# --------------------------------------------------

indices = np.arange(len(df))

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
)

X_text_train = X_text.iloc[train_indices]
X_text_test = X_text.iloc[test_indices]

features_train = feature_df.iloc[train_indices]
features_test = feature_df.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]


# --------------------------------------------------
# 4. Train the improved model
# --------------------------------------------------

text_features = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
)

numeric_features = StandardScaler()

print("\nTraining Model V2...")

# Fit the text vectorizer only on the training split.
X_train_text_matrix = text_features.fit_transform(X_text_train)
X_test_text_matrix = text_features.transform(X_text_test)

# Fit scaling only on training features.
X_train_numeric = numeric_features.fit_transform(features_train)
X_test_numeric = numeric_features.transform(features_test)

from scipy.sparse import csr_matrix, hstack

X_train_combined = hstack([
    X_train_text_matrix,
    csr_matrix(X_train_numeric),
]).tocsr()

X_test_combined = hstack([
    X_test_text_matrix,
    csr_matrix(X_test_numeric),
]).tocsr()

model = Ridge(alpha=10.0)

model.fit(X_train_combined, y_train)

print("Training complete!")


# --------------------------------------------------
# 5. Evaluate Model V2
# --------------------------------------------------

predictions = model.predict(X_test_combined)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n========== MODEL V2 EVALUATION ==========")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")


# --------------------------------------------------
# 6. Save the model and preprocessing components
# --------------------------------------------------

artifact = {
    "model": model,
    "text_vectorizer": text_features,
    "numeric_scaler": numeric_features,
    "feature_columns": list(feature_df.columns),
}

model_path = MODEL_DIR / "resume_match_model_v2.joblib"

joblib.dump(artifact, model_path)

print("\nModel saved to:", model_path)
print("MODEL V2 TRAINING COMPLETE!")
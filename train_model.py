from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


# ---------------------------------------
# 1. Load the dataset
# ---------------------------------------

DATA_PATH = Path("data/job_resume_fit.csv")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)

print("Loading CareerLens dataset...")

df = pd.read_csv(DATA_PATH)

required_columns = [
    "resume_text",
    "job_text",
    "ai_match_score",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")


# ---------------------------------------
# 2. Prepare the data
# ---------------------------------------

df = df.dropna(subset=required_columns).copy()

df["resume_text"] = df["resume_text"].astype(str)
df["job_text"] = df["job_text"].astype(str)
df["ai_match_score"] = pd.to_numeric(
    df["ai_match_score"],
    errors="coerce"
)

df = df.dropna(subset=["ai_match_score"])

# Keep the target within the expected score range.
df = df[
    df["ai_match_score"].between(0, 100)
].copy()

# Combine both texts into one input.
df["combined_text"] = (
    "RESUME: "
    + df["resume_text"]
    + " JOB DESCRIPTION: "
    + df["job_text"]
)

X = df["combined_text"]
y = df["ai_match_score"]

print("Usable records:", len(df))


# ---------------------------------------
# 3. Split training and testing data
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ---------------------------------------
# 4. Build the baseline model
# ---------------------------------------

model = Pipeline([
    (
        "features",
        TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        ),
    ),
    (
        "regressor",
        Ridge(alpha=10.0),
    ),
])


# ---------------------------------------
# 5. Train
# ---------------------------------------

print("\nTraining the model...")

model.fit(X_train, y_train)

print("Training complete!")


# ---------------------------------------
# 6. Evaluate on unseen test records
# ---------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")

print("\n========== SAMPLE PREDICTIONS ==========")

for actual, predicted in zip(
    y_test.iloc[:10],
    predictions[:10],
):
    predicted = max(0, min(100, predicted))

    print(
        f"Actual score: {actual:6.2f} | "
        f"Predicted score: {predicted:6.2f}"
    )


# ---------------------------------------
# 7. Save the trained model
# ---------------------------------------

model_path = MODEL_DIR / "resume_match_model.joblib"

joblib.dump(model, model_path)

print("\nModel saved to:", model_path)
print("CAREERLENS MODEL TRAINING COMPLETE!")
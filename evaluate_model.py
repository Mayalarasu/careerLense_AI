import pandas as pd
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from pathlib import Path


# Load dataset and trained model
df = pd.read_csv("data/job_resume_fit.csv")

model = joblib.load("models/resume_match_model.joblib")

# Prepare data using the same input format as training
df["combined_text"] = (
    "RESUME: "
    + df["resume_text"].astype(str)
    + " JOB DESCRIPTION: "
    + df["job_text"].astype(str)
)

# Reproduce the same test split
_, X_test, _, y_test = train_test_split(
    df["combined_text"],
    df["ai_match_score"],
    test_size=0.20,
    random_state=42,
)

# Keep category information aligned with test records
test_indices = X_test.index
test_df = df.loc[test_indices].copy()

# Predict
test_df["actual_score"] = y_test
test_df["predicted_score"] = model.predict(X_test)

# Overall metrics
print("\n========== OVERALL TEST PERFORMANCE ==========")

mae = mean_absolute_error(
    test_df["actual_score"],
    test_df["predicted_score"],
)

rmse = mean_squared_error(
    test_df["actual_score"],
    test_df["predicted_score"],
) ** 0.5

r2 = r2_score(
    test_df["actual_score"],
    test_df["predicted_score"],
)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

# Category-wise performance
print("\n========== PERFORMANCE BY CATEGORY ==========")

results = []

for category, group in test_df.groupby("category"):
    if len(group) < 2:
        continue

    category_mae = mean_absolute_error(
        group["actual_score"],
        group["predicted_score"],
    )

    category_rmse = mean_squared_error(
        group["actual_score"],
        group["predicted_score"],
    ) ** 0.5

    results.append({
        "category": category,
        "test_records": len(group),
        "MAE": round(category_mae, 2),
        "RMSE": round(category_rmse, 2),
    })

results_df = pd.DataFrame(results)

print(
    results_df.sort_values("MAE").to_string(index=False)
)

# Save the evaluation report
Path("reports").mkdir(exist_ok=True)

results_df.to_csv(
    "reports/category_evaluation.csv",
    index=False,
)

print("\nReport saved to reports/category_evaluation.csv")

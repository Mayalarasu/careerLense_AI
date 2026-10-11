# 🎯 CareerLens AI

### AI-Powered Career Preparation & Resume Skill-Gap Analysis

CareerLens AI is a student-built career preparation application designed to help users understand how their resume aligns with job requirements.

It analyzes resume content, identifies relevant skills, evaluates supporting evidence, highlights potential skill gaps, and suggests learning directions to support career preparation.

---

## 🚀 Features

### 📄 Resume Analysis
- Upload a text-based PDF resume.
- Extract resume text for analysis.
- Identify skills mentioned in the resume.

### 🎯 Skill Matching
- Compare resume skills with job requirements.
- Calculate a skill coverage score.
- Identify matching and missing skills.

### 🔍 Skill Evidence Analysis
- Categorize detected evidence into:
  - Project evidence
  - Work or experience evidence
  - Learning interest
  - Mention only
- Help users understand how their resume demonstrates relevant skills.

### 📚 Learning Recommendations
- Identify potential skill gaps.
- Suggest areas to focus on for further learning.
- Support career preparation through actionable insights.

### 📊 Reports
- Generate downloadable CSV reports containing skill evidence and analysis results.

### 🤖 Machine Learning
- Includes a trained resume-matching model.
- Uses text-based and numerical features to estimate a match score based on the training dataset.

> **Note:** The machine-learning score is based on dataset labels. It is not a hiring probability, a verified measure of proficiency, or a prediction of recruitment outcomes.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Interface:** Streamlit
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas
- **Resume Parsing:** PyMuPDF
- **Data Format:** CSV
- **Model Storage:** Joblib

---

## 📁 Project Structure

```text
CareerLens-AI/
├── app.py
├── career_explorer.py
├── job_templates.py
├── resume_parser.py
├── skill_matcher.py
├── requirements.txt
├── .gitignore
│
├── data/
│   └── job_resume_fit.csv
│
├── models/
│   ├── resume_match_model.joblib
│   └── resume_match_model_v2.joblib
│
├── reports/
│
├── train_model.py
├── train_model_v2.py
├── evaluate_model.py
├── evaluate_skill_overlap.py
├── check_skill_coverage.py
├── expand_skills.py
├── explore_dataset.py
└── inspect_dataset.py

import csv
import io
import re
from difflib import SequenceMatcher
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import hstack, csr_matrix

from resume_parser import extract_text_from_pdf
from skill_matcher import compare_resume_to_job, find_skills, SKILL_DICTIONARY
from job_templates import (
    generate_job_description_for_skill,
    get_popular_skills,
    get_related_roles,
    get_skill_category,
    get_role_title_for_skill,
)


# --------------------------------------------------
# MODEL V2 LOADING AND PREDICTION
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "models" / "resume_match_model_v2.joblib"

@st.cache_resource
def load_model_v2():
    """Load the saved Model V2 artifact once per Streamlit session."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict_model_v2(resume_text, job_description):
    """Predict the dataset-style match score using V2's text and skill features."""
    artifact = load_model_v2()
    if not artifact:
        return None

    resume_skills = set(find_skills(resume_text))
    job_skills = set(find_skills(job_description))
    shared_skills = resume_skills.intersection(job_skills)

    job_skill_count = len(job_skills)
    resume_skill_count = len(resume_skills)
    shared_skill_count = len(shared_skills)

    numeric_values = {
        "resume_skill_count": resume_skill_count,
        "job_skill_count": job_skill_count,
        "shared_skill_count": shared_skill_count,
        "job_skill_overlap": shared_skill_count / job_skill_count if job_skill_count else 0.0,
        "resume_skill_coverage": shared_skill_count / resume_skill_count if resume_skill_count else 0.0,
    }

    feature_columns = artifact.get("feature_columns", list(numeric_values.keys()))
    numeric_frame = pd.DataFrame([[numeric_values[column] for column in feature_columns]],
                                 columns=feature_columns)
    numeric_scaled = artifact["numeric_scaler"].transform(numeric_frame)

    combined_text = f"{resume_text} {job_description}"
    text_vector = artifact["text_vectorizer"].transform([combined_text])
    combined_features = hstack([text_vector, csr_matrix(numeric_scaled)]).tocsr()

    prediction = float(artifact["model"].predict(combined_features)[0])
    # The training labels are dataset scores, so keep the UI score on a 0-100 scale.
    return max(0.0, min(100.0, prediction))


def _normalize_evidence_text(text):
    """Normalize whitespace and remove obvious PDF extraction noise."""
    text = re.sub(r"\s+", " ", str(text)).strip(" ·|—-	")
    return text


def _is_generic_evidence(item):
    """Identify headers, contact details, skill inventories, and generic summaries."""
    text = _normalize_evidence_text(item.get("text", ""))
    lower = text.lower()
    generic_phrases = (
        "professional summary", "additional information", "relevant coursework",
        "coursework:", "programming:", "technical skills", "core competencies",
        "key skills", "tools:", "tools and technologies", "foundations:",
        "data:", "languages:", "frameworks:", "libraries:",
        "alex morgan fictional sample resume", "@example.com",
        "hands-on academic project experience", "currently strengthening",
    )
    if item.get("category") == "Mention only":
        return True
    if any(phrase in lower for phrase in generic_phrases):
        return True
    # A short header-like snippet containing contact/location details is not evidence.
    if "@" in text and len(text) < 220 and not re.search(r"\b(built|developed|implemented|created|trained|managed|tested)\b", lower):
        return True
    return False


def rank_evidence_items(evidence_items):
    """Deduplicate and rank specific, action-oriented evidence above generic text."""
    action_pattern = re.compile(
        r"\b(built|developed|implemented|created|designed|deployed|trained|evaluated|"
        r"tested|cleaned|analyzed|analysed|explored|visualized|visualised|modeled|"
        r"modelled|used|queried|automated|performed|conducted|classified|predicted|"
        r"processed|managed|maintained|configured|integrated|committed|tracked|"
        r"organized|organised|returned|stored|summarized|summarised|maintained)\b",
        re.IGNORECASE,
    )
    ranked = []
    seen = set()
    for item in evidence_items:
        text = _normalize_evidence_text(item.get("text", ""))
        if not text:
            continue
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        cleaned = {**item, "text": text}
        category = cleaned.get("category", "Mention only")
        score = {
            "Project evidence": 30,
            "Work/experience evidence": 25,
            "Learning interest": 5,
            "Mention only": 0,
        }.get(category, 0)
        if action_pattern.search(text):
            score += 20
        if _is_generic_evidence(cleaned):
            score -= 100
        if 60 <= len(text) <= 240:
            score += 8
        elif len(text) < 35:
            score -= 5
        elif len(text) > 320:
            score -= 8
        ranked.append((score, cleaned))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [item for score, item in ranked]


def best_evidence_item(evidence_items):
    """Return the strongest non-generic evidence item, if one exists."""
    ranked = rank_evidence_items(evidence_items)
    return next((item for item in ranked if not _is_generic_evidence(item)), None)


def _evidence_relevant_to_skill(skill, text):
    """Conservatively screen extra snippets for skill-specific relevance.

    The strongest evidence remains untouched. This filter only governs additional
    snippets, so a useful indirect example can still support more than one skill.
    """
    skill_key = re.sub(r"[^a-z0-9]+", " ", str(skill).lower()).strip()
    lower = text.lower()
    concept_groups = {
        # SQL needs evidence of querying/manipulating data with SQL. Merely
        # storing records in SQLite or mentioning a database is not enough.
        "sql": (r"\bsql\b", r"\bqueries?\b", r"\bselect\b", r"\bjoin(?:ed|s)?\b", r"\bgroup by\b", r"\bwhere\b", r"\binsert(?:ed)?\b", r"\bupdate(?:d)?\b", r"\bdelete(?:d)?\b", r"\baggregate[ds]?\b"),
        "rest api": (r"\bapi\b", r"\bhttp\b", r"\bendpoint", r"\bjson\b", r"\bflask\b", r"\brequest", r"\bresponse"),
        "data analysis": (r"\banaly[sz]", r"\bdata\b", r"\bdataset\b", r"\bchart", r"\bvisuali[sz]", r"\btrend", r"\bpattern", r"\bpandas\b", r"\bmissing values\b", r"\bstatistic"),
        "machine learning": (r"\bmachine learning\b", r"\bmodel\b", r"\btrain(?:ed|ing)?\b", r"\bclassif", r"\bpredict", r"\baccuracy\b", r"\bconfusion matrix\b", r"\bfeature"),
        "python": (r"\bpython\b", r"\bscript\b", r"\bpandas\b", r"\bprogram", r"\bcode\b"),
        "pandas": (r"\bpandas\b", r"\bdataframe\b", r"\bdf\.", r"\bread_csv\b", r"\bmissing values\b", r"\bclean(?:ed|ing)?\b"),
        "git": (r"\bgit\b", r"\bversion control\b", r"\bcommit", r"\brepositor", r"\bbranch", r"\bsource code\b"),
    }
    patterns = concept_groups.get(skill_key)
    if not patterns:
        # For other skills, require either the skill phrase or a word from it.
        words = [w for w in skill_key.split() if len(w) > 2]
        return any(re.search(r"\b" + re.escape(w) + r"\b", lower) for w in words)
    return any(re.search(pattern, lower) for pattern in patterns)


def additional_evidence_items(evidence_items, strongest=None, limit=3, exclude_texts=None, skill=None):
    """Return distinct, useful evidence snippets without repeating the best one.

    Uses normalized text and a conservative similarity check to remove near-duplicates
    caused by PDF extraction or overlapping sentence windows. It does not require the
    skill name to appear literally, because indirect project evidence can be relevant.
    """
    ranked = rank_evidence_items(evidence_items)
    excluded = []
    if strongest:
        excluded.append(_normalize_evidence_text(strongest.get("text", "")))

    selected = []
    selected_keys = [
        re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        for text in excluded if text
    ]
    # Do not repeat another requirement's best evidence as an "additional" item.
    # This keeps the report concise while still allowing the same sentence to be
    # the best evidence for more than one genuinely related skill.
    for text in (exclude_texts or []):
        normalized = _normalize_evidence_text(text)
        key = re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()
        if key and key not in selected_keys:
            selected_keys.append(key)

    for item in ranked:
        if _is_generic_evidence(item):
            continue

        text = _normalize_evidence_text(item.get("text", ""))
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if not key:
            continue
        # Exact or near-exact duplicate of the strongest/previous evidence.
        if any(
            key == previous or SequenceMatcher(None, key, previous).ratio() >= 0.88
            for previous in selected_keys
        ):
            continue
        # Reject tiny fragments and section-like labels, but retain concise statements.
        if len(text) < 35 or len(text) > 320:
            continue
        if skill and not _evidence_relevant_to_skill(skill, text):
            continue
        selected.append({**item, "text": text})
        selected_keys.append(key)
        if len(selected) >= limit:
            break
    return selected


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CareerLens AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# DARK UI THEME
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap'
    );

    :root {
        --black: #000000;
        --bg-main: #0a0a0c;
        --bg-card: #121215;
        --bg-surface: #18181c;
        --border-dark: #232328;
        --border-mid: #383842;
        --border-light: #52525e;
        --yellow: #facc15;
        --yellow-bright: #fde047;
        --yellow-dark: #ca8a04;
        --yellow-glow: rgba(250, 204, 21, 0.35);
        --text-white: #ffffff;
        --text-grey-light: #e4e4e7;
        --text-grey-mid: #a1a1aa;
        --text-grey-dark: #71717a;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.5px;
        color: #ffffff;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 15% 0%, rgba(255, 255, 255, 0.045), transparent 45%),
            radial-gradient(ellipse at 85% 15%, rgba(255, 255, 255, 0.035), transparent 40%),
            radial-gradient(ellipse at 50% 65%, rgba(255, 255, 255, 0.025), transparent 50%),
            #0a0a0c;
        color: #ffffff;
        animation: pageReveal 0.65s ease-out both;
    }

    @keyframes pageReveal {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes heroReveal {
        from {
            opacity: 0;
            transform: translateY(18px) scale(0.99);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    @keyframes yellowBorderPulse {
        0%, 100% {
            border-color: rgba(255, 255, 255, 0.30);
            box-shadow: 0 0 20px rgba(255, 255, 255, 0.08);
        }
        50% {
            border-color: rgba(255, 255, 255, 0.55);
            box-shadow: 0 0 28px rgba(255, 255, 255, 0.14);
        }
    }

    [data-testid="stHeader"] {
        background: rgba(10, 10, 12, 0.9);
    }

    /* Hide Streamlit's sidebar toggle */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    [data-testid="stSidebar"] {
        background: #0d0d10;
        border-right: 1px solid #232328;
    }

    [data-testid="stSidebar"] * {
        font-family: 'DM Sans', sans-serif;
    }

    .block-container {
        max-width: 1260px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    /* Black & Yellow Hero */
    .hero {
        position: relative;
        overflow: hidden;
        padding: 42px 38px;
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            #151518 0%,
            #0e0e11 60%,
            #17171b 100%
        );
        border: 1px solid rgba(250, 204, 21, 0.45);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6), 0 0 35px rgba(255, 255, 255, 0.055);
        margin-bottom: 28px;
        animation: heroReveal 0.8s ease-out both;
    }

    .hero::before {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 200px;
        height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.10) 0%, transparent 70%);
        filter: blur(25px);
        pointer-events: none;
    }

    .hero h1 {
        font-size: clamp(32px, 4.2vw, 48px);
        font-weight: 800;
        margin-bottom: 12px;
        line-height: 1.15;
        background: linear-gradient(135deg, #ffffff 40%, #fde047 80%, #facc15 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        max-width: 740px;
        font-size: 15px;
        line-height: 1.8;
        color: #a1a1aa;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 30px;
        background: rgba(250, 204, 21, 0.12);
        border: 1px solid #facc15;
        color: #facc15;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.8px;
        margin-bottom: 18px;
        box-shadow: 0 0 16px rgba(255, 255, 255, 0.08);
    }

    .section-label {
        display: inline-block;
        color: #facc15;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .section-description {
        color: #a1a1aa;
        font-size: 14px;
        margin-bottom: 22px;
        line-height: 1.7;
    }

    /* Black, Yellow & Grey Metric Cards */
    .metric-card {
        border-radius: 20px;
        padding: 22px;
        min-height: 125px;
        margin-bottom: 14px;
        background: #141417;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .metric-card.metric-violet,
    .metric-card.metric-yellow {
        background: linear-gradient(145deg, #1c1c22 0%, #111114 100%);
        border: 1px solid rgba(250, 204, 21, 0.5);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4), 0 0 20px rgba(250, 204, 21, 0.15);
    }
    .metric-card.metric-violet:hover,
    .metric-card.metric-yellow:hover {
        border-color: #facc15;
        box-shadow: 0 12px 35px rgba(250, 204, 21, 0.3);
    }

    .metric-card.metric-emerald,
    .metric-card.metric-white {
        background: linear-gradient(145deg, #1c1c22 0%, #111114 100%);
        border: 1px solid rgba(255, 255, 255, 0.35);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    }
    .metric-card.metric-emerald:hover,
    .metric-card.metric-white:hover {
        border-color: #ffffff;
        box-shadow: 0 12px 35px rgba(255, 255, 255, 0.25);
    }

    .metric-card.metric-cyan,
    .metric-card.metric-grey {
        background: linear-gradient(145deg, #18181c 0%, #111113 100%);
        border: 1px solid #3f3f46;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    }
    .metric-card.metric-cyan:hover,
    .metric-card.metric-grey:hover {
        border-color: #facc15;
        box-shadow: 0 12px 35px rgba(250, 204, 21, 0.2);
    }

    .metric-label {
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1px;
        color: #a1a1aa;
        margin-bottom: 10px;
        text-transform: uppercase;
    }

    .metric-value-violet,
    .metric-value-yellow {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 34px;
        font-weight: 800;
        color: #facc15;
        text-shadow: 0 0 20px rgba(250, 204, 21, 0.4);
    }

    .metric-value-emerald,
    .metric-value-white {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 34px;
        font-weight: 800;
        color: #ffffff;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.35);
    }

    .metric-value-cyan,
    .metric-value-grey {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 34px;
        font-weight: 800;
        color: #fde047;
    }

    .metric-note {
        color: #71717a;
        font-size: 12px;
        margin-top: 8px;
    }

    /* Skill Status Cards */
    .skill-card {
        border-radius: 14px;
        padding: 14px 18px;
        margin: 10px 0;
        background: #141417;
        border: 1px solid #27272a;
        color: #ffffff;
        font-weight: 600;
        display: flex;
        align-items: center;
        transition: all 0.25s ease;
        animation: heroReveal 0.4s ease-out both;
    }

    .skill-card.skill-matched {
        background: linear-gradient(135deg, rgba(250, 204, 21, 0.08) 0%, #131316 100%);
        border: 1px solid rgba(250, 204, 21, 0.45);
        color: #ffffff;
    }
    .skill-card.skill-matched:hover {
        transform: translateX(6px);
        border-color: #facc15;
        box-shadow: 0 6px 20px rgba(250, 204, 21, 0.25);
    }

    .skill-card.skill-weak {
        background: linear-gradient(135deg, rgba(161, 161, 170, 0.08) 0%, #131316 100%);
        border: 1px solid #52525e;
        color: #e4e4e7;
    }
    .skill-card.skill-weak:hover {
        transform: translateX(6px);
        border-color: #a1a1aa;
        box-shadow: 0 6px 20px rgba(161, 161, 170, 0.2);
    }

    .skill-card.skill-missing {
        background: #101013;
        border: 1px solid #2e2e36;
        color: #a1a1aa;
    }
    .skill-card.skill-missing:hover {
        transform: translateX(6px);
        border-color: #71717a;
    }

    .skill-dot {
        display: inline-block;
        height: 10px;
        width: 10px;
        border-radius: 50%;
        margin-right: 12px;
    }

    .matched-dot {
        background: #facc15;
        box-shadow: 0 0 12px #facc15;
    }

    .weak-dot {
        background: #e4e4e7;
        box-shadow: 0 0 10px #e4e4e7;
    }

    .missing-dot {
        background: #52525b;
        box-shadow: 0 0 8px #52525b;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 14px;
        border: 1px solid #2e2e36;
        background: #16161a;
        color: #ffffff;
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        min-height: 44px;
        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        border-color: #facc15;
        color: #facc15;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(250, 204, 21, 0.25);
        background: #1c1c22;
    }

    /* Primary CTA Button (Black & Yellow High-Impact) */
    button[kind="primary"] {
        border: 1px solid #fde047 !important;
        background: linear-gradient(135deg, #facc15 0%, #eab308 100%) !important;
        color: #000000 !important;
        font-size: 16px !important;
        font-weight: 900 !important;
        letter-spacing: 0.8px !important;
        min-height: 52px !important;
        box-shadow: 0 0 22px rgba(255, 255, 255, 0.12), 0 8px 25px rgba(0, 0, 0, 0.7) !important;
    }

    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #fde047 0%, #facc15 100%) !important;
        color: #000000 !important;
        transform: translateY(-3px) scale(1.008) !important;
        box-shadow: 0 0 28px rgba(255, 255, 255, 0.18), 0 12px 30px rgba(0, 0, 0, 0.8) !important;
    }

    .stDownloadButton > button {
        background: #16161a;
        border: 1px solid rgba(250, 204, 21, 0.5);
        color: #ffffff;
        border-radius: 12px;
        min-height: 46px;
        font-weight: 700;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3);
        transition: all 0.25s ease;
    }

    .stDownloadButton > button:hover {
        border-color: #facc15;
        color: #facc15;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(250, 204, 21, 0.25);
    }

    .stTextArea textarea {
        background: #101013 !important;
        color: #ffffff !important;
        border: 1px solid #27272a !important;
        border-radius: 14px !important;
        font-family: 'DM Sans', sans-serif !important;
        line-height: 1.7 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5) !important;
        transition: all 0.25s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: #facc15 !important;
        box-shadow: 0 0 22px rgba(255, 255, 255, 0.12) !important;
    }

    [data-testid="stFileUploader"] {
        background: #101013;
        border: 2px dashed #383842;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        transition: all 0.25s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #facc15;
        background: #141418;
        box-shadow: 0 0 25px rgba(250, 204, 21, 0.2);
    }

    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(
            90deg,
            #facc15 0%,
            #fde047 50%,
            #eab308 100%
        ) !important;
        height: 10px;
        border-radius: 6px;
        box-shadow: 0 0 15px rgba(250, 204, 21, 0.5);
    }

    [data-testid="stExpander"] {
        background: #121215;
        border: 1px solid #27272a;
        border-radius: 14px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.4);
        transition: all 0.25s ease;
    }

    [data-testid="stExpander"]:hover {
        border-color: #facc15;
        box-shadow: 0 6px 24px rgba(250, 204, 21, 0.15);
    }

    /* Target Card & Chips */
    .target-card {
        background: linear-gradient(
            135deg,
            #17171b 0%,
            #0f0f12 60%,
            #141418 100%
        );
        border: 1px solid rgba(250, 204, 21, 0.5);
        border-radius: 20px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.6), 0 0 25px rgba(255, 255, 255, 0.055);
        animation: heroReveal 0.4s ease-out both;
    }

    .role-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        background: #facc15;
        border: 1px solid #fde047;
        color: #000000;
        font-size: 12px;
        font-weight: 800;
        box-shadow: 0 0 12px rgba(250, 204, 21, 0.35);
    }

    .category-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        background: #27272a;
        border: 1px solid #3f3f46;
        color: #ffffff;
        font-size: 12px;
        font-weight: 700;
    }

    .pill-label {
        font-size: 13px;
        font-weight: 800;
        color: #e4e4e7;
        margin-bottom: 8px;
    }

    .meta-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 14px;
        background: #18181c;
        border: 1px solid rgba(250, 204, 21, 0.45);
        color: #facc15;
        font-size: 12px;
        font-weight: 700;
        margin: 3px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }

    /* Hide Streamlit's built-in toolbar: theme switcher, Deploy and menu. */
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu {
        visibility: hidden !important;
        display: none !important;
        height: 0 !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    .footer {
        text-align: center;
        color: var(--muted);
        font-size: 12px;
        line-height: 1.8;
        padding: 24px 0 5px;
    }

    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
        }
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }

        .hero {
            padding: 23px;
        }

        .metric-value {
            font-size: 26px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🎯 CareerLens")
    st.caption("AI-powered career preparation")

    st.divider()

    st.markdown("### Explore")
    st.caption("Use Skill Search to explore skills, then scroll down to analyze a resume against a job description.")
    st.divider()
    st.caption("Built as a student career-preparation project.")


# --------------------------------------------------
# HERO
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">
            ✦ YOUR PERSONAL CAREER COMPANION
        </div>
        <h1>Build a career that<br>moves you forward.</h1>
        <p>
            Understand your resume, discover potential skill gaps,
            and turn job requirements into actionable learning goals.
            Career preparation should be accessible to everyone.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# STEP 01 — TARGET SKILL & OPPORTUNITY SETUP
# --------------------------------------------------

st.markdown(
    '<div class="section-label">01 — TARGET SKILL & CAREER PROFILE</div>',
    unsafe_allow_html=True,
)

st.header("Choose a skill or paste a job description")
st.markdown(
    """
    <div class="section-description">
        Select any skill from our 180+ skill library to <strong>automatically populate a realistic, tailored job description</strong>,
        or paste your own job posting. Then upload your resume to discover your evidence match!
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize session state variables
if "chosen_skill" not in st.session_state:
    st.session_state["chosen_skill"] = None

if "job_description" not in st.session_state:
    st.session_state["job_description"] = ""

skill_names = sorted(SKILL_DICTIONARY.keys(), key=str.casefold)
popular_skills = get_popular_skills()

# Popular skill quick-select buttons
st.markdown('<div class="pill-label">⚡ Quick-Select Popular Career Skills:</div>', unsafe_allow_html=True)
pill_cols = st.columns(len(popular_skills))
for idx, p_skill in enumerate(popular_skills):
    with pill_cols[idx]:
        is_active = (st.session_state.get("chosen_skill") == p_skill)
        label = f"✓ {p_skill}" if is_active else p_skill
        if st.button(label, key=f"quick_pill_{p_skill}", use_container_width=True):
            st.session_state["chosen_skill"] = p_skill
            st.session_state["job_description"] = generate_job_description_for_skill(p_skill)
            st.session_state["skill_search_box"] = p_skill
            st.rerun()

# Searchable full skill dropdown
current_skill = st.session_state.get("chosen_skill")

def on_dropdown_skill_change():
    selected = st.session_state.get("skill_search_box")
    if selected:
        st.session_state["chosen_skill"] = selected
        st.session_state["job_description"] = generate_job_description_for_skill(selected)

if "skill_search_box" not in st.session_state:
    st.session_state["skill_search_box"] = current_skill

chosen_from_search = st.selectbox(
    "Or search CareerLens's complete 180+ skill dictionary:",
    skill_names,
    key="skill_search_box",
    on_change=on_dropdown_skill_change,
    placeholder="Search any skill (e.g., Python, Machine Learning, React, AWS, Docker, Financial Analysis)...",
)

# Display Target Information Card if skill is selected
if current_skill:
    category = get_skill_category(current_skill)
    role_title = get_role_title_for_skill(current_skill)
    related_roles = get_related_roles(current_skill)

    st.markdown(
        f"""
        <div class="target-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span class="role-badge">🎯 Primary Target: {current_skill}</span>
                <span class="category-badge">Domain: {category}</span>
            </div>
            <div style="font-size: 15px; font-weight: 700; color: #fffaf0; margin-bottom: 6px;">
                Typical Role: {role_title}
            </div>
            <div style="font-size: 12px; color: var(--muted); margin-bottom: 6px;">
                <strong>Related career paths:</strong> {' · '.join(related_roles[:4])}
            </div>
            <div style="font-size: 11px; color: #8bd3c7;">
                ✨ The job description below has been automatically filled for <strong>{current_skill}</strong>. You can freely edit it or paste a custom description.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# STEP 02 — RESUME & JOB DETAILS
# --------------------------------------------------

left_col, right_col = st.columns(2, gap="large")

with left_col:
    st.markdown("#### 📄 1. Upload Your Resume")
    uploaded_resume = st.file_uploader(
        "Upload text-based PDF resume",
        type=["pdf"],
        help="Use a text-based PDF. Scanned PDFs may require OCR.",
    )

    if uploaded_resume:
        st.success(f"✓ Resume loaded: **{uploaded_resume.name}** ({round(uploaded_resume.size / 1024, 1)} KB)")
    else:
        st.info("💡 Upload your PDF resume to compare against the target requirements.")

with right_col:
    # Header with live detected skills counter and reset button
    col_hdr, col_btn = st.columns([3, 1])
    with col_hdr:
        st.markdown("#### 📝 2. Target Job Description")
    with col_btn:
        if st.button("🗑️ Clear", key="clear_desc_btn", use_container_width=True):
            st.session_state["job_description"] = ""
            st.session_state["chosen_skill"] = None
            st.rerun()

    # Calculate recognized skills from current text
    current_job_text = st.session_state.get("job_description", "")
    recognized_in_job = find_skills(current_job_text) if current_job_text.strip() else []

    if recognized_in_job:
        chips_html = "".join([f'<span class="meta-tag">{s}</span>' for s in recognized_in_job[:8]])
        if len(recognized_in_job) > 8:
            chips_html += f'<span class="meta-tag">+{len(recognized_in_job) - 8} more</span>'
        st.markdown(
            f'<div style="margin-bottom: 8px;"><small style="color: #8bd3c7; font-weight: 600;">🎯 {len(recognized_in_job)} Recognized Job Requirements:</small><br>{chips_html}</div>',
            unsafe_allow_html=True,
        )

    job_description = st.text_area(
        "Job description",
        key="job_description",
        height=240,
        placeholder="Choose a skill above to auto-fill this box, or paste your target job description here...",
        label_visibility="collapsed",
    )

st.markdown("<br>", unsafe_allow_html=True)

analyze = st.button(
    "✦ Analyze My Career Match",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

if analyze:

    if uploaded_resume is None:
        st.error("Please upload your resume first.")

    elif not job_description.strip():
        st.error("Please paste a job description.")

    else:
        with st.spinner("Analyzing your resume and skill evidence..."):
            resume_text, error = extract_text_from_pdf(uploaded_resume)
            with st.expander("DEBUG: Extracted Resume Text"):
                st.text(resume_text)

        if error:
            st.error(error)

        else:
            results = compare_resume_to_job(
                resume_text,
                job_description,
            )
            with st.expander("DEBUG: Skill Matcher Output"):
                 from skill_matcher import find_skill_evidence

                 st.write("Extracted resume skills:")
                 st.write(find_skills(resume_text))

                 st.write("Recognized job skills:")
                 st.write(results["required_skills"])

                 st.write("Evidence classifications:")
                 st.json(find_skill_evidence(resume_text))

                 st.write("Matched skills:")
                 st.write(results["matched_skills"])
            try:
                results["model_v2_score"] = predict_model_v2(
                    resume_text,
                    job_description,
                )
            except Exception as exc:
                results["model_v2_score"] = None
                st.warning(
                    "Model V2 could not run, so the evidence-based skill analysis "
                    f"is still available. Details: {exc}"
                )

            if not results["required_skills"]:
                st.warning(
                    "No supported skills were recognized in the job "
                    "description. Try a description containing Python, "
                    "SQL, Git, React, or another supported skill."
                )
                st.session_state.pop("analysis", None)

            else:
                st.session_state["analysis"] = results


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

results = st.session_state.get("analysis")

if results:

    st.divider()

    st.markdown(
        '<div class="section-label">02 — YOUR RESULTS</div>',
        unsafe_allow_html=True,
    )

    st.header("Your career match overview")

    st.markdown(
        """
        <div class="section-description">
            Review the skills identified in your resume and the evidence
            supporting each match.
        </div>
        """,
        unsafe_allow_html=True,
    )

    weak_skills = results.get("weak_evidence_skills", [])
    matched_skills = results.get("matched_skills", [])
    missing_skills = results.get("missing_skills", [])
    required_skills = results.get("required_skills", [])
    resume_skills = results.get("resume_skills", [])
    evidence = results.get("resume_evidence", {})
    recommendations = results.get("recommendations", [])

    metric1, metric2, metric3 = st.columns(3)
    with metric1:
        model_score = results.get("model_v2_score")

        st.markdown(
            f"""
            <div class="metric-card metric-yellow">
                <div class="metric-label">⚡ EXPERIMENTAL ML MODEL ESTIMATE</div>
                <div class="metric-value-yellow">
                    {f"{model_score:.1f} / 100" if model_score is not None else "Unavailable"}
                </div>
                <div class="metric-note">
                    Dataset-based estimate, not a hiring probability
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric2:
        st.markdown(
            f"""
            <div class="metric-card metric-white">
                <div class="metric-label">✓ REQUIREMENTS WITH EVIDENCE</div>
                <div class="metric-value-white">
                    {len(matched_skills)} / {len(required_skills)}
                </div>
                <div class="metric-note">
                    Requirements with supporting project/work evidence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric3:
        st.markdown(
            f"""
            <div class="metric-card metric-grey">
                <div class="metric-label">🎯 RECOGNIZED REQUIREMENTS</div>
                <div class="metric-value-grey">{len(required_skills)}</div>
                <div class="metric-note">
                    Supported skills identified in the job description
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("**Evidence-backed requirement coverage**")

    coverage = (
        len(matched_skills) / len(required_skills) * 100
        if required_skills
        else 0
    )

    st.progress(max(0.0, min(1.0, coverage / 100)))

    st.caption(
        f"{coverage:.1f}% = {len(matched_skills)} of "
        f"{len(required_skills)} recognized job skills had at least one "
        "snippet classified as project or work/experience evidence. "
        "This is a heuristic, not a measure of proficiency or hiring probability."
    )

    if results["match_score"] < 50:
        st.warning(
            "Evidence is limited: review the resume manually and ask the candidate "
            "to clarify relevant projects or work before drawing conclusions."
        )
    elif results["match_score"] < 80:
        st.info(
            "Some requirements have supporting project/work evidence, while others "
            "may need clarification or additional examples."
        )

    if results.get("model_v2_score") is not None:
        st.markdown("**Model-predicted dataset score**")
        st.progress(max(0.0, min(1.0, results["model_v2_score"] / 100)))
        st.caption(
            "This separate estimate is learned from the dataset's existing score labels. "
            "It is not validated against real hiring outcomes."
        )
    else:
        st.caption(
            "Model V2 is unavailable. The evidence-backed requirement coverage above "
            "is still available."
        )

    st.divider()

    st.markdown(
        '<div class="section-label">03 — SKILL GAP INSIGHTS</div>',
        unsafe_allow_html=True,
    )

    st.header("What we found")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### ⚡ Evidence detected")

        if matched_skills:
            for skill in matched_skills:
                st.markdown(
                    f"""
                    <div class="skill-card skill-matched">
                        <span class="skill-dot matched-dot"></span>
                        {skill}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No strong project/work evidence detected.")

        st.markdown("#### ◻ Needs verification")

        if weak_skills:
            for skill in weak_skills:
                st.markdown(
                    f"""
                    <div class="skill-card skill-weak">
                        <span class="skill-dot weak-dot"></span>
                        {skill}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No weak-evidence matches identified.")

    with col2:
        st.markdown("#### ✕ Not found in resume")

        if missing_skills:
            for skill in missing_skills:
                st.markdown(
                    f"""
                    <div class="skill-card skill-missing">
                        <span class="skill-dot missing-dot"></span>
                        {skill}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("All recognized requirements appear in the resume.")

        st.markdown("#### ✦ Recognized resume skills")

        if resume_skills:
            st.write(" · ".join(resume_skills))
        else:
            st.caption("No supported skills detected.")

    st.divider()

    st.markdown(
        '<div class="section-label">04 — EXPLAINABLE ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    st.header("Review your evidence")

    st.markdown(
        """
        <div class="section-description">
            Open a skill to see the resume text that triggered its classification.
            Review these results before making career decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    best_by_skill = {
        skill: best_evidence_item(evidence.get(skill, []))
        for skill in required_skills
    }
    all_best_evidence_texts = [
        item["text"] for item in best_by_skill.values() if item
    ]

    for skill in required_skills:
        evidence_items = rank_evidence_items(evidence.get(skill, []))
        strongest = best_by_skill.get(skill)
        other_best_evidence = [
            item["text"] for other_skill, item in best_by_skill.items()
            if item and other_skill != skill
        ]
        additional_items = additional_evidence_items(
            evidence_items, strongest, limit=3, exclude_texts=other_best_evidence, skill=skill
        )

        with st.expander(f"🔎 {skill} — View evidence"):
            if strongest:
                st.markdown("**Best supporting evidence**")
                st.caption(f"Evidence type: {strongest['category']}")
                st.write(f"> {strongest['text']}")

                if additional_items:
                    st.markdown("**Additional relevant evidence**")
                    for item in additional_items[:3]:
                        st.caption(f"Evidence type: {item['category']}")
                        st.write(f"> {item['text']}")
            else:
                st.info("No specific project/work evidence was found. Review the mentions below.")
                mentions = [item for item in evidence_items if _is_generic_evidence(item)]
                if mentions:
                    for item in mentions[:3]:
                        st.caption(f"Evidence type: {item['category']}")
                        st.write(f"> {item['text']}")
                else:
                    st.write("No supporting text was found in the resume.")

    st.divider()

    st.markdown(
        '<div class="section-label">05 — YOUR NEXT STEPS</div>',
        unsafe_allow_html=True,
    )

    st.header("Your learning roadmap")

    if recommendations:
        for item in recommendations:
            with st.expander(f"✦ {item['skill']}"):

                st.write(item["suggestion"])

                st.caption(
                    "A missing or weak keyword is not proof that you lack "
                    "this skill. Verify your experience first."
                )
    else:
        st.success(
            "No missing or weak-evidence skills were identified. "
            "Review the job requirements manually as well."
        )

    st.divider()

    st.markdown(
        '<div class="section-label">06 — TAKE YOUR REPORT</div>',
        unsafe_allow_html=True,
    )

    st.header("Download your analysis")

    report_buffer = io.StringIO()
    writer = csv.writer(report_buffer)

    writer.writerow(["CareerLens AI Analysis"])
    writer.writerow(["Evidence-backed requirement coverage percentage (heuristic)", results["match_score"]])
    writer.writerow(["Model-predicted dataset score (not hiring probability)", results.get("model_v2_score", "Unavailable")])
    writer.writerow([])

    writer.writerow([
        "Required Skill",
        "Status",
        "Best Supporting Evidence",
        "Evidence Type",
        "Additional Relevant Evidence",
    ])

    best_by_skill = {
        skill: best_evidence_item(evidence.get(skill, []))
        for skill in required_skills
    }

    for skill in required_skills:
        if skill in matched_skills:
            status = "Evidence detected (heuristic)"
        elif skill in weak_skills:
            status = "Needs verification"
        else:
            status = "Not found"

        ranked_items = rank_evidence_items(evidence.get(skill, []))
        strongest = best_by_skill.get(skill)
        best_text = strongest["text"] if strongest else "No specific project/work evidence found"
        evidence_type = strongest["category"] if strongest else "Mention only / none"
        other_best_evidence = [
            item["text"] for other_skill, item in best_by_skill.items()
            if item and other_skill != skill
        ]
        additional_items = additional_evidence_items(
            ranked_items, strongest, limit=3, exclude_texts=other_best_evidence, skill=skill
        )
        additional_text = " | ".join(
            f"{item['category']}: {item['text']}"
            for item in additional_items
        )

        writer.writerow([skill, status, best_text, evidence_type, additional_text])

    st.download_button(
        "↓ Download Evidence Report (CSV)",
        data=report_buffer.getvalue(),
        file_name="careerlens_evidence_report.csv",
        mime="text/csv",
        use_container_width=True,
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div class="footer">
        <strong>🎯 CareerLens AI</strong><br>
        Designed and developed by Mayalarasu, a B.Tech AI &amp; Machine Learning student.<br>
        This project brings together resume analysis, explainable skill-gap insights,
        learning recommendations and skill discovery in one place.<br><br>
        CareerLens AI is a student-built career-preparation tool. Results are preliminary
        guidance, not verified qualifications or predictions of hiring outcomes.
    </div>
    """,
    unsafe_allow_html=True,
)   
"""Career Explorer data access and standard job-profile generation.

The bundled profiles are illustrative templates, not live vacancies or employer-specific
job descriptions. They use the skills already supported by CareerLens AI wherever possible.
"""
from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

PROFILE_PATH = Path(__file__).parent / "data" / "career_profiles.json"
EXPERIENCE_LEVELS = (
    "Student / Fresher",
    "Entry Level",
    "Intermediate",
    "Experienced",
)


@lru_cache(maxsize=1)
def load_profiles() -> list[dict[str, Any]]:
    """Load and minimally validate the bundled career profiles."""
    with PROFILE_PATH.open("r", encoding="utf-8") as handle:
        profiles = json.load(handle)

    if not isinstance(profiles, list):
        raise ValueError("Career profile library must be a JSON list.")

    seen_ids: set[str] = set()
    required_fields = {
        "id", "category", "title", "levels", "summary", "responsibilities",
        "required_skills", "optional_skills", "suggested_projects",
    }
    for index, profile in enumerate(profiles):
        if not isinstance(profile, dict) or not required_fields.issubset(profile):
            raise ValueError(f"Career profile at index {index} is missing required fields.")
        if profile["id"] in seen_ids:
            raise ValueError(f"Duplicate career profile ID: {profile['id']}")
        seen_ids.add(profile["id"])
        if not profile["required_skills"]:
            raise ValueError(f"Profile {profile['id']} has no required skills.")
    return profiles


def get_categories() -> list[str]:
    """Return career categories in the order represented in the profile library."""
    categories = dict.fromkeys(profile["category"] for profile in load_profiles())
    return list(categories)


def get_profiles_for_category(category: str) -> list[dict[str, Any]]:
    """Return profiles for one category, sorted by role title."""
    return sorted(
        (p for p in load_profiles() if p["category"] == category),
        key=lambda p: p["title"].casefold(),
    )


def get_profile_by_id(profile_id: str) -> dict[str, Any] | None:
    """Find one profile by its stable ID."""
    return next((p for p in load_profiles() if p["id"] == profile_id), None)



def get_required_skills_for_level(profile: dict[str, Any], experience_level: str) -> list[str]:
    """Return the configured core requirements for the selected experience level."""
    if experience_level not in EXPERIENCE_LEVELS:
        raise ValueError(f"Unsupported experience level: {experience_level}")
    required = list(profile["required_skills"])
    # Keep the fresher profile focused on the three most foundational requirements.
    return required[:3] if experience_level == "Student / Fresher" else required


def build_job_description(
    profile: dict[str, Any],
    experience_level: str,
) -> str:
    """Create an editable job-description draft from a selected standard profile.

    Optional skills are intentionally omitted from the generated description because
    the current matcher treats all recognized skills in the description as requirements.
    The UI can show optional skills separately without inflating the coverage denominator.
    """
    if experience_level not in EXPERIENCE_LEVELS:
        raise ValueError(f"Unsupported experience level: {experience_level}")

    required_skills = get_required_skills_for_level(profile, experience_level)
    if experience_level == "Student / Fresher":
        level_guidance = (
            "Level focus: foundational understanding, coursework, guided practice, "
            "and beginner-friendly projects. Equivalent learning evidence may be relevant."
        )
    elif experience_level == "Entry Level":
        level_guidance = (
            "Level focus: core skills, supervised responsibilities, practical learning, "
            "and clear communication."
        )
    elif experience_level == "Intermediate":
        level_guidance = (
            "Level focus: independent delivery, practical problem-solving, quality, "
            "and coordination with others."
        )
    else:
        level_guidance = (
            "Level focus: ownership of complex work, sound judgment, quality, and "
            "collaboration. Actual seniority expectations vary by employer."
        )

    lines = [
        f"{profile['title']} — {experience_level}",
        "STANDARD CAREER PROFILE — illustrative template, not a live vacancy",
        "",
        "Job Summary:",
        profile["summary"],
        "",
        "Typical Responsibilities:",
    ]
    lines.extend(f"- {item}" for item in profile["responsibilities"])
    lines.extend([
        "",
        "Core Skills for This Profile:",
        *[f"- {skill}" for skill in required_skills],
        "",
        level_guidance,
        "",
        "Note: Adapt this profile to the actual job advertisement and verify any required "
        "qualifications, licenses, certifications, and experience.",
    ])
    return "\n".join(lines)


def profile_count_by_category() -> dict[str, int]:
    """Return the number of profiles per category for UI/testing."""
    counts: dict[str, int] = defaultdict(int)
    for profile in load_profiles():
        counts[profile["category"]] += 1
    return dict(counts)

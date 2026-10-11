import re


# ============================================================
# CAREERLENS AI — MULTI-CAREER SKILL MATCHER
# Supports skills across 23 career categories.
# Rule-based prototype; detected skills are not verified claims.
# ============================================================


SKILL_DICTIONARY = {

    # 1. Information Technology
    "Python": ["python"],
    "SQL": ["sql", "sql query", "sql queries", "sqlite", "mysql", "postgresql", "database query", "database queries"],
    "Java": ["java"],
    "JavaScript": ["javascript"],
    "C++": ["c++"],
    "C": ["c programming", "c language"],
    "React": ["react", "react.js"],
    "HTML": ["html"],
    "CSS": ["css", "cascading style sheets"],
    "REST API": ["rest api", "restful api", "api development"],
    "Git": ["git", "github", "gitlab"],
    "Linux": ["linux"],
    "Docker": ["docker"],
    "Cloud Computing": ["cloud computing", "cloud platforms"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["microsoft azure", "azure cloud"],
    "Cybersecurity": ["cybersecurity", "cyber security", "information security"],
    "Networking": ["computer networking", "network administration"],
    "IT Infrastructure": ["it infrastructure", "server management"],
   "Database Management": [
    "database management",
    "database systems",
    "client databases",
    "maintained databases",
    "maintained client databases",
    "database records",
    "maintained client databases and records",
],
    "Data Structures and Algorithms": ["data structures", "algorithms", "dsa"],

    # 2. AI and Machine Learning
    "Machine Learning": ["machine learning", "scikit-learn", "sklearn", "machine-learning model", "machine learning model", "ml model", "classification model", "machine-learning classification model"],
    "Deep Learning": ["deep learning", "neural networks"],
    "Natural Language Processing": ["natural language processing", "nlp"],
    "Computer Vision": ["computer vision", "opencv"],
    "Data Analysis": ["data analysis", "data analytics", "data cleaning", "exploratory data analysis", "analyzed data", "analysed data", "analyze data", "analyse data", "analyzed the data", "analysed the data", "analyze the data", "analyse the data", "created charts to explain the results", "created charts to explain results", "charts to explain the results", "cleaned and explored", "clean and explore", "explore missing values", "data exploration", "explored missing values", "summarize results", "summarised results", "summarized results", "summarised the results", "created simple charts", "patterns found in the dataset", "analyzed datasets", "analysed datasets", "analyze datasets", "analyse datasets", "analyzed operational data", "analysed operational data", "analyze operational data", "analyse operational data", "operational data analysis", "generated charts", "prepared charts", "communicate findings"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Statistics": ["statistical analysis", "statistics"],
    "Data Visualization": ["data visualization", "data visualisation"],
    "Power BI": ["power bi"],
    "Tableau": ["tableau"],

    # 3. Business Development
    "Business Development": ["business development"],
    "Market Research": ["market research"],
    "Business Strategy": ["business strategy", "strategic planning"],
    "Lead Generation": ["lead generation"],
    "Client Relationship Management": ["client relationship management"],
    "Negotiation": ["negotiation", "negotiation skills"],
    "Sales Strategy": ["sales strategy"],

    # 4. Chef and Food Services
    "Food Preparation": ["food preparation"],
    "Food Safety": ["food safety", "food hygiene"],
    "Menu Planning": ["menu planning"],
    "Kitchen Management": ["kitchen management"],
    "Inventory Management": ["inventory management"],
    "Culinary Skills": ["culinary skills", "culinary arts"],
    "Baking": ["baking", "pastry preparation"],

    # 5. Finance
    "Financial Analysis": ["financial analysis"],
    "Financial Reporting": ["financial reporting"],
    "Budgeting": ["budgeting", "budget preparation"],
    "Forecasting": ["financial forecasting", "forecasting"],
    "Risk Management": ["risk management"],
    "Investment Analysis": ["investment analysis"],
    "Financial Modeling": ["financial modeling", "financial modelling"],
    "Taxation": ["taxation", "tax accounting"],
    "Compliance": ["regulatory compliance", "compliance management"],
    "Excel": ["microsoft excel", "excel spreadsheets"],

    # 6. Engineering
    "AutoCAD": ["autocad"],
    "CAD": ["computer-aided design", "cad software"],
    "Quality Control": ["quality control", "quality assurance"],
    "Project Engineering": ["project engineering"],
    "Mechanical Design": ["mechanical design"],
    "Electrical Systems": ["electrical systems"],
    "Troubleshooting": ["troubleshooting"],
    "Technical Documentation": ["technical documentation"],

    # 7. Accounting
    "Accounting": ["accounting", "accounting operations"],
    "Bookkeeping": ["bookkeeping"],
    "Auditing": ["auditing", "audit coordination"],
    "Accounts Payable": ["accounts payable"],
    "Accounts Receivable": ["accounts receivable"],
    "Reconciliation": ["account reconciliation", "bank reconciliation"],
    "GAAP": ["gaap"],
    "ERP Systems": ["erp systems", "enterprise resource planning"],

    # 8. Advocate and Legal
    "Legal Research": ["legal research"],
    "Legal Writing": ["legal writing"],
    "Contract Law": ["contract law"],
    "Litigation": ["litigation"],
    "Case Management": ["case management"],
    "Legal Compliance": ["legal compliance"],
    "Client Counseling": ["client counseling", "client counselling"],

    # 9. Fitness
    "Personal Training": ["personal training", "personal trainer"],
    "Fitness Assessment": ["fitness assessment"],
    "Exercise Programming": ["exercise programming", "workout programming"],
    "Strength Training": ["strength training"],
    "Nutrition Basics": ["nutrition planning", "sports nutrition"],
    "First Aid": ["first aid"],
    "CPR": ["cpr", "cardiopulmonary resuscitation"],

    # 10. Sales
    "Sales": ["sales experience", "sales operations"],
    "Customer Service": ["customer service", "customer support"],
    "CRM Software": ["crm software", "salesforce"],
    "Prospecting": ["sales prospecting", "prospecting"],
    "Account Management": ["account management"],
    "Sales Negotiation": ["sales negotiation"],

    # 11. Aviation
    "Flight Safety": ["flight safety", "aviation safety"],
    "Flight Planning": ["flight planning"],
    "Air Traffic Control": ["air traffic control"],
    "Aircraft Maintenance": ["aircraft maintenance"],
    "Aviation Regulations": ["aviation regulations"],
    "Navigation": ["aircraft navigation", "aviation navigation"],

    # 12. Healthcare
    "Patient Care": ["patient care"],
    "Clinical Assessment": ["clinical assessment"],
    "Patient Monitoring": ["patient monitoring"],
    "Medical Records": ["medical records", "electronic health records"],
    "Medication Administration": ["medication administration"],
    "Infection Control": ["infection control"],
    "Basic Life Support": ["basic life support", "bls"],
    "Advanced Cardiac Life Support": [
        "advanced cardiac life support",
        "acls",
    ],
    "Nursing": ["nursing", "nursing care"],
    "Clinical Documentation": ["clinical documentation"],

    # 13. Consulting
    "Problem Solving": ["problem solving", "problem-solving"],
    "Strategic Planning": ["strategic planning"],
    "Stakeholder Management": ["stakeholder management"],
    "Process Improvement": ["process improvement"],
    "Requirements Analysis": ["requirements analysis"],
    "Change Management": ["change management"],

    # 14. Banking
    "Banking Operations": ["banking operations"],
    "Credit Analysis": ["credit analysis"],
    "Loan Processing": ["loan processing"],
    "KYC": ["know your customer", "kyc"],
    "AML": ["anti-money laundering", "aml"],
    "Fraud Detection": ["fraud detection"],
    "Retail Banking": ["retail banking"],

    # 15. Construction
    "Construction Management": ["construction management"],
    "Site Supervision": ["site supervision"],
    "Building Codes": ["building codes", "building regulations"],
    "Structural Engineering": ["structural engineering"],
    "Quantity Surveying": ["quantity surveying"],
    "Project Scheduling": ["project scheduling"],
    "Health and Safety": [
        "occupational health and safety",
        "site safety",
    ],

    # 16. Public Relations
    "Public Relations": ["public relations", "pr campaigns"],
    "Media Relations": ["media relations"],
    "Press Releases": ["press releases", "press release writing"],
    "Crisis Communication": ["crisis communication"],
    "Communication Skills": ["communication skills"],
    "Event Planning": ["event planning"],

    # 17. Human Resources
    "Recruitment": ["recruitment", "talent acquisition"],
    "Interviewing": ["interviewing", "candidate interviews"],
    "Onboarding": ["employee onboarding", "onboarding"],
    "Payroll": ["payroll processing", "payroll"],
    "Employee Relations": ["employee relations"],
    "HR Policies": ["human resources policies", "hr policies"],
    "Performance Management": ["performance management"],
    "HRIS": ["human resources information system", "hris"],

    # 18. Designer
    "Graphic Design": ["graphic design"],
    "UI/UX Design": [
        "ui/ux",
        "user experience design",
        "user interface design",
    ],
    "Figma": ["figma"],
    "Adobe Photoshop": ["adobe photoshop", "photoshop"],
    "Adobe Illustrator": ["adobe illustrator"],
    "Wireframing": ["wireframing", "wireframes"],
    "Prototyping": ["prototyping", "interactive prototypes"],
    "Typography": ["typography"],

    # 19. Arts
    "Creative Writing": ["creative writing"],
    "Illustration": ["illustration"],
    "Photography": ["photography"],
    "Video Editing": ["video editing"],
    "Animation": ["animation"],
    "Digital Illustration": ["digital illustration"],
    "Visual Storytelling": ["visual storytelling"],

    # 20. Teaching
    "Lesson Planning": ["lesson planning"],
    "Classroom Management": ["classroom management"],
    "Curriculum Development": ["curriculum development"],
    "Student Assessment": ["student assessment"],
    "Educational Technology": ["educational technology"],
    "Differentiated Instruction": ["differentiated instruction"],
    "Teaching": ["teaching experience", "classroom teaching"],

    # 21. Digital Media
    "SEO": [
        "search engine optimization",
        "search engine optimisation",
        "seo",
    ],
    "Content Marketing": ["content marketing"],
    "Social Media Marketing": ["social media marketing"],
    "Google Analytics": ["google analytics"],
    "Email Marketing": ["email marketing"],
    "Content Creation": ["content creation"],
    "Digital Advertising": ["digital advertising", "paid advertising"],
    "Campaign Analytics": ["campaign analytics"],

    # 22. Agriculture
    "Crop Management": ["crop management"],
    "Soil Management": ["soil management"],
    "Irrigation": ["irrigation management", "irrigation"],
    "Agronomy": ["agronomy"],
    "Pest Management": ["pest management"],
    "Agricultural Machinery": ["agricultural machinery"],
    "Sustainable Agriculture": ["sustainable agriculture"],

    # 23. Automobile
    "Automotive Diagnostics": [
        "automotive diagnostics",
        "vehicle diagnostics",
    ],
    "Vehicle Maintenance": ["vehicle maintenance"],
    "Engine Repair": ["engine repair"],
    "Automotive Electronics": ["automotive electronics"],
    "Automotive Systems": ["automotive systems"],
    "Automotive Troubleshooting": ["automotive troubleshooting"],
}


# ============================================================
# LEARNING RECOMMENDATIONS
# ============================================================

LEARNING_RESOURCES = {
    "Python": "Practise Python functions, collections, and file handling.",
    "SQL": "Build a database project using joins and aggregations.",
    "Machine Learning": "Build a small model and evaluate it on held-out data.",
    "Data Analysis": "Practise cleaning and visualizing data with Pandas.",
    "Git": "Practise commits, branches, merges, and GitHub workflows.",
    "REST API": "Build a REST API and test its endpoints.",
    "JavaScript": "Practise JavaScript fundamentals and asynchronous code.",
    "React": "Build a React application with components and state.",
    "C++": "Practise C++ fundamentals and object-oriented programming.",
    "Java": "Build a Java application using classes.",
    "Deep Learning": "Study neural networks and build a small classifier.",
    "Computer Vision": "Experiment with image processing and classification.",
    "Natural Language Processing": "Practise text preprocessing and classification.",
    "HTML": "Create a semantic web page.",
    "CSS": "Build and style a responsive web page.",
    "Cloud Computing": "Explore deployment using a free cloud tier.",
    "Docker": "Containerize a simple Python application.",
    "Linux": "Practise shell commands and file permissions.",
    "Financial Analysis": "Analyse a sample company's financial statements.",
    "Financial Reporting": "Practise preparing financial statements.",
    "Budgeting": "Create a personal or business budget in a spreadsheet.",
    "Accounting": "Practise journal entries and financial statements.",
    "Auditing": "Study audit procedures and internal controls.",
    "Patient Care": "Study appropriate patient-care fundamentals.",
    "Clinical Assessment": "Study clinical assessment principles with qualified guidance.",
    "Nursing": "Use accredited nursing education resources.",
    "Recruitment": "Practise writing job descriptions and screening criteria.",
    "Lesson Planning": "Create a lesson plan with learning objectives.",
    "Classroom Management": "Study classroom management strategies.",
    "Graphic Design": "Create a small design portfolio.",
    "UI/UX Design": "Design and test a simple user interface.",
    "Figma": "Create a prototype using Figma.",
    "SEO": "Learn keyword research and on-page SEO fundamentals.",
    "Content Marketing": "Create a content plan for a sample brand.",
    "AutoCAD": "Practise technical drawing using suitable CAD software.",
    "Construction Management": "Study construction planning and safety procedures.",
    "Agronomy": "Study crop science through reputable agricultural resources.",
    "Aircraft Maintenance": "Use approved aviation training and safety resources.",
    "Automotive Diagnostics": "Practise diagnostics with appropriate training and supervision.",
}


# ============================================================
# EVIDENCE CLASSIFICATION
# ============================================================

# These patterns are intentionally heuristic. They identify language
# suggesting evidence; they do not verify that a skill was truly used.
INTEREST_PATTERNS = [
    r"\binterested in learning\b",
    r"\bwant to learn\b",
    r"\bhope to learn\b",
    r"\bplanning to learn\b",
    r"\bbeginner (?:in|at|with)\b",
    r"\bcurrently learning\b",
    r"\blearning about\b",
    r"\blooking to learn\b",
    r"\baspiring\b",
]

# Action verbs commonly used to describe hands-on technical projects.
PROJECT_ACTION_PATTERNS = [
    r"\bbuilt\b",
    r"\bdeveloped\b",
    r"\bimplemented\b",
    r"\bcreated\b",
    r"\bdesigned\b",
    r"\bdeployed\b",
    r"\bprogrammed\b",
    r"\btrained\b",
    r"\bevaluated\b",
    r"\btested\b",
    r"\bcleaned\b",
    r"\banaly[sz]ed\b",
    r"\banaly[sz]ing\b",
    r"\bexplored\b",
    r"\bexploratory\b",
    r"\bvisuali[sz]ed\b",
    r"\bmodeled\b",
    r"\bmodelled\b",
    r"\bused\b",
    r"\bwrote\b",
    r"\bwritten\b",
    r"\bcalculated\b",
    r"\bsummarized\b",
    r"\bsummarised\b",
    r"\bqueried\b",
    r"\bautomated\b",
    r"\bperformed\b",
    r"\bconducted\b",
    r"\bclassified\b",
    r"\bpredicted\b",
    r"\bprocessed\b",
    r"\bmanaged\b",
    r"\bmaintained\b",
    r"\bconfigured\b",
    r"\bintegrated\b",
    r"\bcommitted\b",
    r"\btracked\b",
    r"\borganized\b",
    r"\borganised\b",
    r"\bmaintained\b",
    r"\bmanaged\b",
]

EXPERIENCE_PATTERNS = [
    r"\bprofessional experience\b",
    r"\bwork experience\b",
    r"\binternship\b",
    r"\bemployment\b",
    r"\bworked (?:as|on|with|at|for)\b",
    r"\bprofessional background\b",
    r"\bjob role\b",
    r"\bresponsibilit(?:y|ies)\b",
]

# If a skill occurs in one of these sections, a skills list alone should
# not be promoted to project/work evidence.
SKILLS_SECTION_PATTERNS = [
    r"^\s*(?:technical\s+)?skills\s*:?\s*$",
    r"^\s*technical\s+skills\s*[:\-]",
    r"^\s*core\s+competencies\s*[:\-]",
    r"^\s*key\s+skills\s*[:\-]",
    r"^\s*technologies\s*(?:used)?\s*[:\-]",
    r"^\s*tools\s*(?:and\s+technologies)?\s*[:\-]",
    r"^\s*(?:programming|foundations|data|languages|frameworks|libraries|coursework|relevant coursework|certifications)\s*[:\-]",
]

def _contains_alias(text, alias):
    """Match an alias without matching it inside unrelated words."""
    pattern = (
        r"(?<!\w)"
        + re.escape(alias.strip().lower())
        + r"(?!\w)"
    )
    return re.search(pattern, text.lower()) is not None


def _split_resume_text(text):
    """Split resume text into evidence units while retaining section context."""
    raw_lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in str(text).replace("\r", "\n").splitlines()
        if line.strip()
    ]
    if not raw_lines:
        return []

    heading_re = re.compile(
        r"^(?:"
       r"summary|career profile|profile|objective|education|experience|work experience|"
        r"adult care experience|childcare experience|professional experience|employment history|"
        r"professional experience|employment history|projects?|technical skills|skills|"
        r"core competencies|key skills|programming|foundations|data|languages|frameworks|"
        r"libraries|coursework|relevant coursework|certifications|tools(?: and technologies)?|"
        r"technologies(?: used)?|learning interests|professional development"
        r")\s*:?\s*$",
        re.IGNORECASE,
    )
    labeled_list_re = re.compile(
        r"^(?:technical\s+skills|skills|core\s+competencies|key\s+skills|"
        r"programming|foundations|data|languages|frameworks|libraries|"
        r"coursework|relevant coursework|certifications|tools(?: and technologies)?|"
        r"technologies(?: used)?)\s*[:\-]",
        re.IGNORECASE,
    )

    def section_for_heading(heading, current):
        h = heading.lower().strip().rstrip(":")
        if h in {"work experience", "professional experience", "experience",
                 "employment history", "professional background"}:
            return "work"
        if h in {"project", "projects"}:
            return "project"
        if h in {"skills", "technical skills", "core competencies", "key skills",
                 "programming", "foundations", "data", "languages", "frameworks",
                 "libraries", "coursework", "relevant coursework", "certifications",
                 "tools", "tools and technologies", "technologies", "technologies used"}:
            return "skills"
        if h in {"learning interests", "professional development"}:
            return "learning"
        if h in {"summary", "career profile", "profile", "objective", "education"}:
            return h
        return current

    paragraphs = []
    current = ""
    current_section = "general"
    paragraph_section = current_section

    for line in raw_lines:
        is_heading = bool(heading_re.match(line))
        is_labeled_list = bool(labeled_list_re.match(line))
        is_bullet = bool(re.match(r"^(?:[-•*▪]|\d+[.)])\s*", line))

        if is_heading:
            if current:
                paragraphs.append((current, paragraph_section))
                current = ""
            current_section = section_for_heading(line, current_section)
            paragraphs.append((line, current_section))
            continue

        if is_labeled_list or is_bullet:
            if current:
                paragraphs.append((current, paragraph_section))
                current = ""
            paragraphs.append((line, current_section))
            continue

        if not current:
            current = line
            paragraph_section = current_section
        elif re.search(r"[.!?;:]$", current):
            paragraphs.append((current, paragraph_section))
            current = line
            paragraph_section = current_section
        else:
            current = f"{current} {line}"

    if current:
        paragraphs.append((current, paragraph_section))

    units = []
    for paragraph, section in paragraphs:
        if labeled_list_re.match(paragraph) or heading_re.match(paragraph):
            units.append((paragraph.strip(), section))
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", paragraph)
        units.extend((part.strip(), section) for part in parts if part.strip())
    return units


def _classify_evidence(sentence, section=None):
    """Classify evidence using both wording and its resume section."""
    lower = sentence.lower().strip()

    # Learning intent takes precedence, even if the sentence contains action words.
    if section == "learning" or any(re.search(pattern, lower) for pattern in INTEREST_PATTERNS):
        return "Learning interest"

    # A skills list alone is a mention, not evidence of hands-on use.
    if section == "skills" or any(re.search(pattern, lower) for pattern in SKILLS_SECTION_PATTERNS):
        return "Mention only"

    # Explicit resume section context takes priority over generic action verbs.
    if section == "work":
        return "Work/experience evidence"
    if section == "project":
        return "Project evidence"

    # Work evidence requires specific employment/work phrasing.
    if any(re.search(pattern, lower) for pattern in EXPERIENCE_PATTERNS):
        return "Work/experience evidence"
    if any(re.search(pattern, lower) for pattern in PROJECT_ACTION_PATTERNS):
        return "Project evidence"
    return "Mention only"


def find_skill_evidence(text):
    """Find skill aliases and classify the surrounding text.

    Categories are heuristic labels, not verified claims or proficiency scores.
    """
    if not text:
        return {}

    lines = _split_resume_text(text)
    evidence = {}

    for skill, aliases in SKILL_DICTIONARY.items():
        classifications = []
        seen = set()
        for sentence, section in lines:
            if any(_contains_alias(sentence, alias) for alias in aliases):
                normalized = re.sub(r"\s+", " ", sentence).strip()
                if normalized.lower() in seen:
                    continue
                seen.add(normalized.lower())
                classifications.append({
                    "text": normalized,
                    "category": _classify_evidence(normalized, section),
                })
        if classifications:
            evidence[skill] = classifications
    return evidence


def find_skills(text):
    """Return the recognized skill names."""
    return list(find_skill_evidence(text).keys())


def compare_resume_to_job(resume_text, job_description):
    """
    Compare recognized resume skills with job requirements.

    A requirement is counted as supported if at least one matched resume
    snippet is classified as project or work/experience evidence. This is
    a heuristic and does not verify the truth or quality of a claim.
    """
    resume_evidence = find_skill_evidence(resume_text)
    job_skills = find_skills(job_description)

    matched_skills = []
    weak_evidence_skills = []
    missing_skills = []

    for skill in job_skills:
        evidence_items = resume_evidence.get(skill, [])
        categories = {item["category"] for item in evidence_items}

        if (
            "Project evidence" in categories
            or "Work/experience evidence" in categories
        ):
            matched_skills.append(skill)
        elif evidence_items:
            weak_evidence_skills.append(skill)
        else:
            missing_skills.append(skill)

    match_score = (
        round(len(matched_skills) / len(job_skills) * 100)
        if job_skills else 0
    )

    recommendations = [
        {
            "skill": skill,
            "suggestion": LEARNING_RESOURCES.get(
                skill,
                f"Find reputable learning resources and practise {skill}.",
            ),
        }
        for skill in missing_skills + weak_evidence_skills
    ]

    return {
        "resume_skills": list(resume_evidence.keys()),
        "required_skills": job_skills,
        "matched_skills": matched_skills,
        "weak_evidence_skills": weak_evidence_skills,
        "missing_skills": missing_skills,
        "match_score": match_score,
        "recommendations": recommendations,
        "resume_evidence": resume_evidence,
    }

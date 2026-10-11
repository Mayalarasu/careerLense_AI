"""
CareerLens AI - Intelligent Job Description and Career Template Generator.

Generates realistic, tailored job descriptions for any skill in SKILL_DICTIONARY,
including relevant role titles, responsibilities, and companion requirements.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from skill_matcher import SKILL_DICTIONARY

# -------------------------------------------------------------------------
# SKILL CATEGORIES & DOMAINS
# -------------------------------------------------------------------------

CATEGORY_DEFAULTS = {
    "Information Technology": {
        "title": "Software & Systems Specialist",
        "dept": "Technology & Engineering",
        "resp": [
            "Develop, test, and maintain robust software modules and technical solutions.",
            "Write clean, maintainable code following team best practices and version control.",
            "Collaborate with cross-functional teams to resolve technical challenges and defects.",
            "Participate in system architecture reviews and technical documentation.",
        ],
        "core_companions": ["Git", "Problem Solving", "Linux", "REST API"],
    },
    "AI and Machine Learning": {
        "title": "Machine Learning & AI Specialist",
        "dept": "Data Science & Artificial Intelligence",
        "resp": [
            "Design, train, and validate machine learning and statistical models on real datasets.",
            "Perform data cleaning, exploratory analysis, and feature engineering.",
            "Evaluate model performance using appropriate metrics and error diagnostics.",
            "Document experimental findings and present actionable insights to stakeholders.",
        ],
        "core_companions": ["Python", "Data Analysis", "Pandas", "NumPy", "Statistics", "Git"],
    },
    "Business Development": {
        "title": "Business Development Executive",
        "dept": "Commercial & Growth",
        "resp": [
            "Identify and evaluate new market opportunities, business leads, and strategic partners.",
            "Build and nurture long-term relationships with prospective and current clients.",
            "Develop commercial proposals, pitch presentations, and growth strategies.",
            "Analyze market trends and competitor activities to inform sales positioning.",
        ],
        "core_companions": ["Market Research", "Negotiation", "Sales Strategy", "Client Relationship Management"],
    },
    "Chef and Food Services": {
        "title": "Culinary Specialist / Chef",
        "dept": "Kitchen & Food Operations",
        "resp": [
            "Prepare high-quality culinary offerings adhering strictly to recipes and presentation standards.",
            "Enforce strict kitchen sanitation, hygiene, and food safety regulations at all times.",
            "Manage ingredient inventory, minimize kitchen waste, and assist with menu planning.",
            "Coordinate food service timing with kitchen staff and front-of-house teams.",
        ],
        "core_companions": ["Food Safety", "Food Preparation", "Kitchen Management", "Inventory Management"],
    },
    "Finance": {
        "title": "Financial Analyst & Planning Associate",
        "dept": "Corporate Finance & Treasury",
        "resp": [
            "Conduct financial variance analysis, cost tracking, and revenue forecasting.",
            "Build dynamic financial models and quantitative forecasts to guide business decisions.",
            "Prepare monthly, quarterly, and annual financial summaries for management.",
            "Ensure adherence to corporate fiscal policies and regulatory compliance.",
        ],
        "core_companions": ["Financial Analysis", "Excel", "Budgeting", "Forecasting", "Risk Management"],
    },
    "Engineering": {
        "title": "Project & Design Engineer",
        "dept": "Engineering Operations",
        "resp": [
            "Develop technical designs, engineering calculations, and system specifications.",
            "Perform troubleshooting and root-cause analysis on technical processes and equipment.",
            "Oversee quality control procedures and adhere to industry engineering standards.",
            "Create comprehensive technical documentation, drawings, and testing manuals.",
        ],
        "core_companions": ["CAD", "Quality Control", "Troubleshooting", "Technical Documentation"],
    },
    "Accounting": {
        "title": "Accountant & Financial Records Specialist",
        "dept": "Accounting & Finance",
        "resp": [
            "Manage ledger entries, accounts payable, accounts receivable, and balance reconciliations.",
            "Conduct periodic audit preparations and adhere to standard GAAP accounting principles.",
            "Generate accurate trial balances, profit/loss statements, and balance sheets.",
            "Coordinate with tax and compliance teams for timely reporting.",
        ],
        "core_companions": ["Bookkeeping", "Reconciliation", "Auditing", "Accounts Payable", "GAAP"],
    },
    "Advocate and Legal": {
        "title": "Legal Counsel & Compliance Associate",
        "dept": "Legal & Regulatory Affairs",
        "resp": [
            "Conduct thorough legal research, case precedent reviews, and statutory analyses.",
            "Draft, review, and negotiate commercial agreements, contracts, and legal memorandums.",
            "Provide sound legal counsel on regulatory compliance and mitigate institutional liability.",
            "Assist with dispute resolution, litigation filings, and client advisory.",
        ],
        "core_companions": ["Legal Research", "Legal Writing", "Contract Law", "Legal Compliance"],
    },
    "Fitness": {
        "title": "Fitness Coach & Personal Trainer",
        "dept": "Wellness & Athletic Training",
        "resp": [
            "Conduct initial client fitness assessments, mobility screenings, and health evaluations.",
            "Design customized exercise programs focused on progressive strength and conditioning.",
            "Instruct clients on proper exercise form, injury prevention, and safe equipment usage.",
            "Provide foundational guidance on lifestyle habits and balanced nutritional practices.",
        ],
        "core_companions": ["Personal Training", "Fitness Assessment", "Strength Training", "Exercise Programming"],
    },
    "Sales": {
        "title": "Sales Account Executive",
        "dept": "Sales & Revenue Operations",
        "resp": [
            "Prospect, qualify, and convert new inbound and outbound customer accounts.",
            "Maintain accurate pipeline tracking and opportunity management inside CRM systems.",
            "Conduct product demonstrations and handle objections through consultative negotiation.",
            "Consistently meet or exceed monthly sales targets and retention goals.",
        ],
        "core_companions": ["CRM Software", "Customer Service", "Prospecting", "Sales Negotiation"],
    },
    "Aviation": {
        "title": "Aviation Operations Specialist",
        "dept": "Flight Operations & Safety",
        "resp": [
            "Review flight plans, weather forecasts, route logistics, and navigation data.",
            "Ensure strict adherence to civil aviation regulations and safety management protocols.",
            "Coordinate maintenance inspections and pre-flight airworthiness checklists.",
            "Communicate effectively with ground control, crew members, and air traffic authorities.",
        ],
        "core_companions": ["Flight Safety", "Flight Planning", "Aviation Regulations", "Navigation"],
    },
    "Healthcare": {
        "title": "Clinical Healthcare Specialist",
        "dept": "Patient Care & Clinical Services",
        "resp": [
            "Deliver direct patient care and perform routine clinical assessments and vital monitoring.",
            "Maintain meticulous and confidential electronic medical records and clinical notes.",
            "Follow strict infection control measures and clinical safety guidelines.",
            "Collaborate with multidisciplinary healthcare teams to deliver patient treatment plans.",
        ],
        "core_companions": ["Patient Care", "Clinical Assessment", "Medical Records", "Infection Control"],
    },
    "Consulting": {
        "title": "Management Consultant / Strategy Analyst",
        "dept": "Strategy & Advisory",
        "resp": [
            "Analyze business workflows, client operations, and identify structural bottlenecks.",
            "Conduct stakeholder interviews and gather detailed requirements for business transformation.",
            "Formulate strategic recommendations, executive decks, and change-management plans.",
            "Facilitate client workshops and oversee project milestone delivery.",
        ],
        "core_companions": ["Problem Solving", "Strategic Planning", "Stakeholder Management", "Process Improvement"],
    },
    "Banking": {
        "title": "Banking Operations & Risk Officer",
        "dept": "Retail & Commercial Banking",
        "resp": [
            "Process customer transactions, account openings, and credit/loan applications diligently.",
            "Execute Know Your Customer (KYC) and Anti-Money Laundering (AML) due diligence.",
            "Monitor accounts for unusual transaction patterns and suspicious fraud indicators.",
            "Provide professional banking support while upholding strict financial regulations.",
        ],
        "core_companions": ["Banking Operations", "KYC", "AML", "Credit Analysis", "Fraud Detection"],
    },
    "Construction": {
        "title": "Construction Site & Project Coordinator",
        "dept": "Construction & Field Operations",
        "resp": [
            "Supervise on-site construction activities, contractor crews, and daily build progress.",
            "Ensure full adherence to local building codes, structural plans, and safety protocols.",
            "Track project schedules, material deliveries, and quantity surveying logs.",
            "Conduct safety inspections to maintain zero-incident job sites.",
        ],
        "core_companions": ["Construction Management", "Site Supervision", "Building Codes", "Health and Safety"],
    },
    "Public Relations": {
        "title": "Public Relations & Communications Specialist",
        "dept": "Corporate Communications & PR",
        "resp": [
            "Draft and distribute compelling press releases, media kits, and corporate announcements.",
            "Cultivate strong relations with industry journalists, editors, and media outlets.",
            "Develop proactive crisis communication frameworks and brand reputation messaging.",
            "Coordinate press conferences, media interviews, and special promotional events.",
        ],
        "core_companions": ["Public Relations", "Media Relations", "Press Releases", "Communication Skills"],
    },
    "Human Resources": {
        "title": "Human Resources & Talent Specialist",
        "dept": "People Operations & HR",
        "resp": [
            "Lead end-to-end recruitment cycles: sourcing, screening, interviewing, and offer extension.",
            "Coordinate employee onboarding, orientation, and training programs.",
            "Manage employee relations inquiries, policy interpretations, and conflict resolution.",
            "Maintain HRIS records, track attendance/payroll, and uphold HR compliance.",
        ],
        "core_companions": ["Recruitment", "Interviewing", "Onboarding", "Employee Relations", "HR Policies"],
    },
    "Designer": {
        "title": "UI/UX & Product Designer",
        "dept": "Design & User Experience",
        "resp": [
            "Create user journeys, wireframes, user flows, and interactive digital prototypes.",
            "Conduct user research and usability testing to validate design hypotheses.",
            "Design polished, accessible user interfaces following established design systems.",
            "Collaborate closely with frontend engineers to ensure design fidelity.",
        ],
        "core_companions": ["UI/UX Design", "Figma", "Wireframing", "Prototyping", "Graphic Design"],
    },
    "Arts": {
        "title": "Creative Visual Artist & Media Creator",
        "dept": "Creative Studio & Media",
        "resp": [
            "Produce original creative illustrations, visual artwork, and digital assets.",
            "Conceptualize visual themes, storyboards, and creative treatments for client projects.",
            "Edit visual and multimedia content to ensure narrative coherence and aesthetic appeal.",
            "Maintain cohesive artistic styles across varied marketing and publication deliverables.",
        ],
        "core_companions": ["Illustration", "Digital Illustration", "Visual Storytelling", "Photography"],
    },
    "Teaching": {
        "title": "Educator & Instructional Specialist",
        "dept": "Academic & Instruction",
        "resp": [
            "Develop structured lesson plans, instructional materials, and learning assessments.",
            "Foster an engaging, inclusive, and disciplined classroom learning environment.",
            "Utilize educational technologies to enrich student comprehension and practical skills.",
            "Evaluate student performance and provide constructive, differentiated feedback.",
        ],
        "core_companions": ["Lesson Planning", "Classroom Management", "Curriculum Development", "Student Assessment"],
    },
    "Digital Media": {
        "title": "Digital Marketing & Growth Specialist",
        "dept": "Marketing & Digital Channels",
        "resp": [
            "Plan and execute data-driven digital marketing campaigns across organic and paid channels.",
            "Optimize content and web properties for Search Engine Optimization (SEO) ranking.",
            "Track and evaluate campaign performance metrics using Google Analytics and dashboards.",
            "Develop high-converting email sequences and engaging social media marketing content.",
        ],
        "core_companions": ["SEO", "Content Marketing", "Social Media Marketing", "Google Analytics"],
    },
    "Agriculture": {
        "title": "Agronomist & Farm Operations Specialist",
        "dept": "Agricultural Science & Production",
        "resp": [
            "Oversee crop growth cycles, soil testing, nutrient applications, and irrigation regimes.",
            "Implement Integrated Pest Management (IPM) to protect crop health sustainably.",
            "Operate and maintain modern agricultural machinery and precision farming tools.",
            "Monitor yields and champion sustainable, environmentally sound farming practices.",
        ],
        "core_companions": ["Crop Management", "Soil Management", "Irrigation", "Pest Management"],
    },
    "Automobile": {
        "title": "Automotive Diagnostics & Maintenance Technician",
        "dept": "Automotive Engineering & Service",
        "resp": [
            "Perform comprehensive diagnostic scans and mechanical troubleshooting on vehicles.",
            "Inspect and repair engine systems, brakes, transmissions, and electrical circuits.",
            "Execute scheduled vehicle servicing and preventive maintenance procedures.",
            "Document diagnostic findings and ensure compliance with automotive safety standards.",
        ],
        "core_companions": ["Troubleshooting", "Quality Control", "Technical Documentation"],
    },
}

# -------------------------------------------------------------------------
# SKILL TO CATEGORY LOOKUP (Populated from SKILL_DICTIONARY structure)
# -------------------------------------------------------------------------

SKILL_TO_CATEGORY: Dict[str, str] = {
    # 1. Information Technology
    "Python": "Information Technology",
    "SQL": "Information Technology",
    "Java": "Information Technology",
    "JavaScript": "Information Technology",
    "C++": "Information Technology",
    "C": "Information Technology",
    "React": "Information Technology",
    "HTML": "Information Technology",
    "CSS": "Information Technology",
    "REST API": "Information Technology",
    "Git": "Information Technology",
    "Linux": "Information Technology",
    "Docker": "Information Technology",
    "Cloud Computing": "Information Technology",
    "AWS": "Information Technology",
    "Azure": "Information Technology",
    "Cybersecurity": "Information Technology",
    "Networking": "Information Technology",
    "IT Infrastructure": "Information Technology",
    "Database Management": "Information Technology",
    "Data Structures and Algorithms": "Information Technology",

    # 2. AI and Machine Learning
    "Machine Learning": "AI and Machine Learning",
    "Deep Learning": "AI and Machine Learning",
    "Natural Language Processing": "AI and Machine Learning",
    "Computer Vision": "AI and Machine Learning",
    "Data Analysis": "AI and Machine Learning",
    "Pandas": "AI and Machine Learning",
    "NumPy": "AI and Machine Learning",
    "TensorFlow": "AI and Machine Learning",
    "PyTorch": "AI and Machine Learning",
    "Statistics": "AI and Machine Learning",
    "Data Visualization": "AI and Machine Learning",
    "Power BI": "AI and Machine Learning",
    "Tableau": "AI and Machine Learning",

    # 3. Business Development
    "Business Development": "Business Development",
    "Market Research": "Business Development",
    "Business Strategy": "Business Development",
    "Lead Generation": "Business Development",
    "Client Relationship Management": "Business Development",
    "Negotiation": "Business Development",
    "Sales Strategy": "Business Development",

    # 4. Chef and Food Services
    "Food Preparation": "Chef and Food Services",
    "Food Safety": "Chef and Food Services",
    "Menu Planning": "Chef and Food Services",
    "Kitchen Management": "Chef and Food Services",
    "Inventory Management": "Chef and Food Services",
    "Culinary Skills": "Chef and Food Services",
    "Baking": "Chef and Food Services",

    # 5. Finance
    "Financial Analysis": "Finance",
    "Financial Reporting": "Finance",
    "Budgeting": "Finance",
    "Forecasting": "Finance",
    "Risk Management": "Finance",
    "Investment Analysis": "Finance",
    "Financial Modeling": "Finance",
    "Taxation": "Finance",
    "Compliance": "Finance",
    "Excel": "Finance",

    # 6. Engineering
    "AutoCAD": "Engineering",
    "CAD": "Engineering",
    "Quality Control": "Engineering",
    "Project Engineering": "Engineering",
    "Mechanical Design": "Engineering",
    "Electrical Systems": "Engineering",
    "Troubleshooting": "Engineering",
    "Technical Documentation": "Engineering",

    # 7. Accounting
    "Accounting": "Accounting",
    "Bookkeeping": "Accounting",
    "Auditing": "Accounting",
    "Accounts Payable": "Accounting",
    "Accounts Receivable": "Accounting",
    "Reconciliation": "Accounting",
    "GAAP": "Accounting",
    "ERP Systems": "Accounting",

    # 8. Advocate and Legal
    "Legal Research": "Advocate and Legal",
    "Legal Writing": "Advocate and Legal",
    "Contract Law": "Advocate and Legal",
    "Litigation": "Advocate and Legal",
    "Case Management": "Advocate and Legal",
    "Legal Compliance": "Advocate and Legal",
    "Client Counseling": "Advocate and Legal",

    # 9. Fitness
    "Personal Training": "Fitness",
    "Fitness Assessment": "Fitness",
    "Exercise Programming": "Fitness",
    "Strength Training": "Fitness",
    "Nutrition Basics": "Fitness",
    "First Aid": "Fitness",
    "CPR": "Fitness",

    # 10. Sales
    "Sales": "Sales",
    "Customer Service": "Sales",
    "CRM Software": "Sales",
    "Prospecting": "Sales",
    "Account Management": "Sales",
    "Sales Negotiation": "Sales",

    # 11. Aviation
    "Flight Safety": "Aviation",
    "Flight Planning": "Aviation",
    "Air Traffic Control": "Aviation",
    "Aircraft Maintenance": "Aviation",
    "Aviation Regulations": "Aviation",
    "Navigation": "Aviation",

    # 12. Healthcare
    "Patient Care": "Healthcare",
    "Clinical Assessment": "Healthcare",
    "Patient Monitoring": "Healthcare",
    "Medical Records": "Healthcare",
    "Medication Administration": "Healthcare",
    "Infection Control": "Healthcare",
    "Basic Life Support": "Healthcare",
    "Advanced Cardiac Life Support": "Healthcare",
    "Nursing": "Healthcare",
    "Clinical Documentation": "Healthcare",

    # 13. Consulting
    "Problem Solving": "Consulting",
    "Strategic Planning": "Consulting",
    "Stakeholder Management": "Consulting",
    "Process Improvement": "Consulting",
    "Requirements Analysis": "Consulting",
    "Change Management": "Consulting",

    # 14. Banking
    "Banking Operations": "Banking",
    "Credit Analysis": "Banking",
    "Loan Processing": "Banking",
    "KYC": "Banking",
    "AML": "Banking",
    "Fraud Detection": "Banking",
    "Retail Banking": "Banking",

    # 15. Construction
    "Construction Management": "Construction",
    "Site Supervision": "Construction",
    "Building Codes": "Construction",
    "Structural Engineering": "Construction",
    "Quantity Surveying": "Construction",
    "Project Scheduling": "Construction",
    "Health and Safety": "Construction",

    # 16. Public Relations
    "Public Relations": "Public Relations",
    "Media Relations": "Public Relations",
    "Press Releases": "Public Relations",
    "Crisis Communication": "Public Relations",
    "Communication Skills": "Public Relations",
    "Event Planning": "Public Relations",

    # 17. Human Resources
    "Recruitment": "Human Resources",
    "Interviewing": "Human Resources",
    "Onboarding": "Human Resources",
    "Payroll": "Human Resources",
    "Employee Relations": "Human Resources",
    "HR Policies": "Human Resources",
    "Performance Management": "Human Resources",
    "HRIS": "Human Resources",

    # 18. Designer
    "Graphic Design": "Designer",
    "UI/UX Design": "Designer",
    "Figma": "Designer",
    "Adobe Photoshop": "Designer",
    "Adobe Illustrator": "Designer",
    "Wireframing": "Designer",
    "Prototyping": "Designer",
    "Typography": "Designer",

    # 19. Arts
    "Creative Writing": "Arts",
    "Illustration": "Arts",
    "Photography": "Arts",
    "Video Editing": "Arts",
    "Animation": "Arts",
    "Digital Illustration": "Arts",
    "Visual Storytelling": "Arts",

    # 20. Teaching
    "Lesson Planning": "Teaching",
    "Classroom Management": "Teaching",
    "Curriculum Development": "Teaching",
    "Student Assessment": "Teaching",
    "Educational Technology": "Teaching",
    "Differentiated Instruction": "Teaching",
    "Teaching": "Teaching",

    # 21. Digital Media
    "SEO": "Digital Media",
    "Content Marketing": "Digital Media",
    "Social Media Marketing": "Digital Media",
    "Google Analytics": "Digital Media",
    "Email Marketing": "Digital Media",
    "Content Creation": "Digital Media",
    "Digital Advertising": "Digital Media",
    "Campaign Analytics": "Digital Media",

    # 22. Agriculture
    "Crop Management": "Agriculture",
    "Soil Management": "Agriculture",
    "Irrigation": "Agriculture",
    "Agronomy": "Agriculture",
    "Pest Management": "Agriculture",
    "Agricultural Machinery": "Agriculture",
    "Sustainable Agriculture": "Agriculture",

    # 23. Automobile
    "Automotive Repair": "Automobile",
    "Vehicle Diagnostics": "Automobile",
    "Engine Maintenance": "Automobile",
    "Brake Systems": "Automobile",
    "Electrical Diagnostics": "Automobile",
    "Preventive Maintenance": "Automobile",
}

# Specific role title mappings for individual skills
CUSTOM_ROLE_TITLES: Dict[str, str] = {
    "Python": "Python Developer / Backend Engineer",
    "SQL": "Database Developer & SQL Analyst",
    "Java": "Java Software Engineer",
    "JavaScript": "Full-Stack / JavaScript Developer",
    "React": "Frontend Developer (React.js)",
    "HTML": "Web Developer & UI Specialist",
    "CSS": "Web Designer & Frontend Stylist",
    "REST API": "Backend API Developer",
    "Git": "Software Versioning & Collaboration Specialist",
    "Linux": "Linux Systems Administrator",
    "Docker": "DevOps & Containerization Engineer",
    "AWS": "Cloud Infrastructure Engineer (AWS)",
    "Azure": "Cloud Solutions Architect (Azure)",
    "Cybersecurity": "Cybersecurity & InfoSec Analyst",
    "Machine Learning": "Machine Learning Engineer",
    "Deep Learning": "Deep Learning / AI Researcher",
    "Natural Language Processing": "NLP & Language Model Engineer",
    "Computer Vision": "Computer Vision Engineer",
    "Data Analysis": "Data Analyst & Insights Specialist",
    "Pandas": "Data Wrangling & Python Analyst",
    "Power BI": "Power BI & Business Intelligence Analyst",
    "Tableau": "Tableau & Visual Analytics Consultant",
    "Excel": "Data & Financial Analyst (Excel)",
    "Financial Analysis": "Financial Analyst",
    "Financial Modeling": "Financial Modeling Specialist",
    "Accounting": "Corporate Accountant",
    "Auditing": "Internal / Financial Auditor",
    "Taxation": "Tax Associate & Compliance Specialist",
    "AutoCAD": "CAD / AutoCAD Drafter & Engineer",
    "UI/UX Design": "UI/UX Product Designer",
    "Figma": "Product Designer (Figma Specialist)",
    "Graphic Design": "Senior Graphic Designer",
    "Legal Research": "Legal Research Associate",
    "Recruitment": "Talent Acquisition Specialist",
    "SEO": "SEO & Organic Growth Specialist",
    "Content Marketing": "Content Marketing Specialist",
    "Social Media Marketing": "Social Media Strategist",
    "Public Relations": "Public Relations Specialist",
    "Patient Care": "Registered Nurse / Healthcare Provider",
    "Personal Training": "Certified Personal Trainer",
}

# -------------------------------------------------------------------------
# GENERATOR FUNCTION
# -------------------------------------------------------------------------

def get_skill_category(skill: str) -> str:
    """Return the career category for a given skill."""
    return SKILL_TO_CATEGORY.get(skill, "Information Technology")


def get_role_title_for_skill(skill: str) -> str:
    """Return an intuitive professional role title for a skill."""
    if skill in CUSTOM_ROLE_TITLES:
        return CUSTOM_ROLE_TITLES[skill]
    category = get_skill_category(skill)
    category_defaults = CATEGORY_DEFAULTS.get(category, {})
    default_title = category_defaults.get("title", f"{skill} Specialist")
    return f"{skill} {default_title.split()[-1]}" if skill not in default_title else default_title


def get_companion_skills_for_skill(skill: str, count: int = 4) -> List[str]:
    """
    Select complementary skills from the same category or related domain
    that exist in SKILL_DICTIONARY to create a rich, realistic job requirement set.
    """
    category = get_skill_category(skill)
    # Find all skills in the same category
    same_cat = [s for s, c in SKILL_TO_CATEGORY.items() if c == category and s != skill]

    companions = []
    # If IT / AI, companion skills often cross-pollinate
    if category in ("Information Technology", "AI and Machine Learning"):
        synergies = {
            "Python": ["SQL", "Git", "REST API", "Data Analysis", "Linux"],
            "SQL": ["Python", "Database Management", "Data Analysis", "Git"],
            "React": ["JavaScript", "HTML", "CSS", "REST API", "Git"],
            "Machine Learning": ["Python", "Pandas", "NumPy", "Data Analysis", "Statistics", "Git"],
            "Data Analysis": ["Python", "SQL", "Pandas", "Data Visualization", "Statistics"],
            "AWS": ["Docker", "Linux", "Cloud Computing", "Git", "Networking"],
            "Docker": ["Linux", "AWS", "Git", "Cloud Computing", "REST API"],
            "Cybersecurity": ["Networking", "Linux", "IT Infrastructure", "Cloud Computing"],
            "Java": ["SQL", "REST API", "Git", "Linux", "Data Structures and Algorithms"],
            "JavaScript": ["HTML", "CSS", "React", "REST API", "Git"],
        }
        if skill in synergies:
            return synergies[skill][:count]

    # Defaults from category
    defaults = CATEGORY_DEFAULTS.get(category, {}).get("core_companions", [])
    for d in defaults:
        if d != skill and d in SKILL_DICTIONARY and d not in companions:
            companions.append(d)

    # Fill from remaining same-category skills
    for s in same_cat:
        if s not in companions:
            companions.append(s)
        if len(companions) >= count:
            break

    return companions[:count]


def generate_job_description_for_skill(
    skill: str,
    experience_level: str = "Entry to Intermediate",
) -> str:
    """
    Generate a complete, professional, copy-paste ready job description
    specifically centered around the target skill and its ecosystem.
    """
    category = get_skill_category(skill)
    role_title = get_role_title_for_skill(skill)
    category_info = CATEGORY_DEFAULTS.get(
        category,
        CATEGORY_DEFAULTS["Information Technology"],
    )

    companion_skills = get_companion_skills_for_skill(skill, count=4)
    responsibilities = category_info.get("resp", [])

    # Assemble job description lines
    lines = [
        f"Position: {role_title}",
        f"Department: {category_info.get('dept', category)}",
        f"Experience Level: {experience_level}",
        f"Target Domain: {category}",
        "",
        "Job Summary:",
        (
            f"We are seeking a talented and proactive professional specializing in {skill}. "
            f"In this role, you will apply your expertise in {skill} to drive meaningful project "
            f"results, improve quality and efficiency, and collaborate with multidisciplinary team members."
        ),
        "",
        "Key Responsibilities:",
        f"- Lead and execute core initiatives requiring hands-on proficiency in {skill}.",
    ]

    for resp in responsibilities:
        lines.append(f"- {resp}")

    lines.extend([
        "",
        "Required Skills & Core Qualifications:",
        f"- {skill} (Primary Core Competency)",
    ])

    for comp in companion_skills:
        lines.append(f"- {comp}")

    # General foundational skills recognized by CareerLens
    if "Problem Solving" not in companion_skills and skill != "Problem Solving":
        lines.append("- Problem Solving and critical thinking")

    lines.extend([
        "",
        "Preferred Attributes:",
        "- Proven track record of delivering projects with documented evidence.",
        "- Strong written and verbal communication skills.",
        "- Ability to adapt to fast-paced team priorities.",
    ])

    return "\n".join(lines)


def get_popular_skills() -> List[str]:
    """Curated list of high-demand skills across key industries for quick UI pills."""
    return [
        "Python",
        "Machine Learning",
        "SQL",
        "React",
        "Data Analysis",
        "AWS",
        "Cybersecurity",
        "Docker",
        "UI/UX Design",
        "Financial Analysis",
        "Excel",
        "Git",
    ]


def get_related_roles(skill: str) -> List[str]:
    """Return common role titles associated with a skill."""
    category = get_skill_category(skill)
    primary = get_role_title_for_skill(skill)
    candidates = [primary]

    role_matrix = {
        "Python": ["Python Backend Engineer", "Data Analyst", "Machine Learning Engineer", "Automation Developer"],
        "SQL": ["Database Administrator", "BI Analyst", "Data Engineer", "Financial Analyst"],
        "Machine Learning": ["ML Engineer", "AI Researcher", "Data Scientist", "NLP Specialist"],
        "React": ["Frontend Engineer", "React Developer", "Full-Stack Developer", "Web Application Engineer"],
        "AWS": ["Cloud Solutions Architect", "DevOps Engineer", "Cloud Infrastructure Specialist"],
        "Data Analysis": ["Data Analyst", "Business Intelligence Analyst", "Analytics Consultant", "Product Analyst"],
        "UI/UX Design": ["Product Designer", "UX Researcher", "UI Designer", "Design Systems Lead"],
        "Cybersecurity": ["Information Security Analyst", "SOC Analyst", "Penetration Tester", "Security Engineer"],
        "Financial Analysis": ["Financial Analyst", "FP&A Specialist", "Investment Analyst", "Risk Associate"],
        "Docker": ["DevOps Engineer", "Site Reliability Engineer (SRE)", "Infrastructure Engineer"],
    }

    if skill in role_matrix:
        return role_matrix[skill]

    # Generate from category
    cat_default = CATEGORY_DEFAULTS.get(category, {}).get("title", f"{skill} Specialist")
    if cat_default not in candidates:
        candidates.append(cat_default)
    candidates.append(f"Junior / Associate {skill} Specialist")
    candidates.append(f"Senior {skill} Lead")
    return candidates

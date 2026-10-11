from skill_matcher import SKILL_DICTIONARY

ADDITIONAL_SKILLS = {
    # Finance and accounting
    "Financial Analysis": ["financial analysis"],
    "Financial Reporting": ["financial reporting"],
    "Budgeting": ["budgeting", "budget preparation"],
    "Forecasting": ["financial forecasting"],
    "Investment Analysis": ["investment analysis"],
    "Financial Modeling": ["financial modeling", "financial modelling"],
    "Taxation": ["taxation", "tax accounting"],
    "Accounting": ["accounting", "accounting operations"],
    "Bookkeeping": ["bookkeeping"],
    "Auditing": ["auditing", "audit coordination"],
    "Excel": ["microsoft excel", "excel spreadsheets"],

    # Healthcare
    "Patient Care": ["patient care"],
    "Clinical Assessment": ["clinical assessment"],
    "Patient Monitoring": ["patient monitoring"],
    "Medical Records": ["medical records", "electronic health records"],
    "Medication Administration": ["medication administration"],
    "Infection Control": ["infection control"],
    "Basic Life Support": ["basic life support", "bls"],
    "Nursing": ["nursing", "nursing care"],

    # Human resources
    "Recruitment": ["recruitment", "talent acquisition"],
    "Interviewing": ["interviewing", "candidate interviews"],
    "Onboarding": ["employee onboarding", "onboarding"],
    "Payroll": ["payroll processing", "payroll"],
    "Employee Relations": ["employee relations"],
    "Performance Management": ["performance management"],

    # Teaching
    "Lesson Planning": ["lesson planning"],
    "Classroom Management": ["classroom management"],
    "Curriculum Development": ["curriculum development"],
    "Student Assessment": ["student assessment"],

    # Design and arts
    "Graphic Design": ["graphic design"],
    "UI/UX Design": ["ui/ux", "user experience design", "user interface design"],
    "Figma": ["figma"],
    "Photography": ["photography"],
    "Video Editing": ["video editing"],
    "Animation": ["animation"],

    # Digital marketing
    "SEO": ["search engine optimization", "search engine optimisation", "seo"],
    "Content Marketing": ["content marketing"],
    "Social Media Marketing": ["social media marketing"],
    "Google Analytics": ["google analytics"],
    "Email Marketing": ["email marketing"],

    # Engineering and construction
    "AutoCAD": ["autocad"],
    "CAD": ["computer-aided design", "cad software"],
    "Quality Control": ["quality control", "quality assurance"],
    "Construction Management": ["construction management"],
    "Structural Engineering": ["structural engineering"],
    "Quantity Surveying": ["quantity surveying"],

    # Agriculture
    "Crop Management": ["crop management"],
    "Soil Management": ["soil management"],
    "Irrigation": ["irrigation management", "irrigation"],
    "Agronomy": ["agronomy"],
    "Pest Management": ["pest management"],

    # Aviation and automobile
    "Flight Safety": ["flight safety", "aviation safety"],
    "Flight Planning": ["flight planning"],
    "Air Traffic Control": ["air traffic control"],
    "Aircraft Maintenance": ["aircraft maintenance"],
    "Automotive Diagnostics": ["automotive diagnostics", "vehicle diagnostics"],
    "Vehicle Maintenance": ["vehicle maintenance"],
    "Engine Repair": ["engine repair"],

    # Sales, business, consulting and communication
    "Business Development": ["business development"],
    "Market Research": ["market research"],
    "Lead Generation": ["lead generation"],
    "Customer Service": ["customer service", "customer support"],
    "Account Management": ["account management"],
    "Strategic Planning": ["strategic planning"],
    "Stakeholder Management": ["stakeholder management"],
    "Process Improvement": ["process improvement"],
    "Public Relations": ["public relations", "pr campaigns"],
    "Media Relations": ["media relations"],
    "Press Releases": ["press releases", "press release writing"],
    "Event Planning": ["event planning"],

    # Banking and legal
    "Banking Operations": ["banking operations"],
    "Credit Analysis": ["credit analysis"],
    "Loan Processing": ["loan processing"],
    "KYC": ["know your customer", "kyc"],
    "AML": ["anti-money laundering", "aml"],
    "Legal Research": ["legal research"],
    "Legal Writing": ["legal writing"],
    "Contract Law": ["contract law"],
    "Litigation": ["litigation"],

    # Food services and fitness
    "Food Preparation": ["food preparation"],
    "Food Safety": ["food safety", "food hygiene"],
    "Menu Planning": ["menu planning"],
    "Kitchen Management": ["kitchen management"],
    "Personal Training": ["personal training", "personal trainer"],
    "Fitness Assessment": ["fitness assessment"],
    "Strength Training": ["strength training"],
}

# Add the new entries without replacing existing ones.
for skill, aliases in ADDITIONAL_SKILLS.items():
    if skill not in SKILL_DICTIONARY:
        SKILL_DICTIONARY[skill] = aliases

# Generate a separate file so we can verify the expanded dictionary.
import pprint

output = (
    "SKILL_DICTIONARY = "
    + pprint.pformat(SKILL_DICTIONARY, sort_dicts=False)
    + "\n"
)

with open("expanded_skill_dictionary.py", "w", encoding="utf-8") as file:
    file.write(output)

print("Original skill count:", 18)
print("Expanded skill count:", len(SKILL_DICTIONARY))
print("Financial Analysis added:", "Financial Analysis" in SKILL_DICTIONARY)
print("Budgeting added:", "Budgeting" in SKILL_DICTIONARY)
print("Generated expanded_skill_dictionary.py")
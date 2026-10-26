"""Feature schema used by the app, the API and the MCP server. No heavy imports on purpose."""

FEATURES = ['HighBP', 'HighChol', 'BMI', 'GenHlth', 'Age', 'Income', 'Education']

BINARY_FEATURES = ("HighBP", "HighChol")
CATEGORY_RANGES = {"GenHlth": (1, 5), "Age": (1, 13), "Income": (1, 8), "Education": (1, 6)}
BMI_RANGE = (12.0, 98.0)

# labels follow the CDC BRFSS 2015 codebook
YES_NO = {0: "No", 1: "Yes"}
GENERAL_HEALTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
AGE_GROUPS = {
    1: "18 to 24", 2: "25 to 29", 3: "30 to 34", 4: "35 to 39", 5: "40 to 44",
    6: "45 to 49", 7: "50 to 54", 8: "55 to 59", 9: "60 to 64", 10: "65 to 69",
    11: "70 to 74", 12: "75 to 79", 13: "80 or older",
}
EDUCATION = {
    1: "Never attended school or only kindergarten",
    2: "Grades 1 through 8 (elementary)",
    3: "Grades 9 through 11 (some high school)",
    4: "Grade 12 or GED (high school graduate)",
    5: "College 1 to 3 years (some college or technical school)",
    6: "College 4 years or more (college graduate)",
}
INCOME = {
    1: "Less than $10,000",
    2: "$10,000 to less than $15,000",
    3: "$15,000 to less than $20,000",
    4: "$20,000 to less than $25,000",
    5: "$25,000 to less than $35,000",
    6: "$35,000 to less than $50,000",
    7: "$50,000 to less than $75,000",
    8: "$75,000 or more",
}

LABELS = {
    "HighBP": YES_NO, "HighChol": YES_NO, "GenHlth": GENERAL_HEALTH,
    "Age": AGE_GROUPS, "Education": EDUCATION, "Income": INCOME,
}

DESCRIPTIONS = {
    "HighBP": "Told by a health professional that blood pressure is high",
    "HighChol": "Told by a health professional that blood cholesterol is high",
    "BMI": "Body Mass Index in kg/m2",
    "GenHlth": "Self-rated general health",
    "Age": "Age group in 5-year bands",
    "Income": "Annual household income band in US dollars",
    "Education": "Highest school grade completed",
}

SHORT_NAMES = {
    "HighBP": "High blood pressure", "HighChol": "High cholesterol", "BMI": "BMI",
    "GenHlth": "General health", "Age": "Age group", "Income": "Household income",
    "Education": "Education",
}

RISK_LABELS = {0: "lower-risk pattern", 1: "higher-risk pattern"}
RISK_MEANING = (
    "1 means the answers resemble survey respondents who reported prediabetes or diabetes, "
    "0 means they resemble respondents who did not. It is a screening-style pattern, not a "
    "diagnosis, and no probability is provided."
)


def get_feature_schema() -> dict:
    """Input schema and output meaning of the risk model, as plain JSON-friendly data."""
    features = []
    for name in FEATURES:
        entry = {"name": name, "description": DESCRIPTIONS[name]}
        if name in LABELS:
            entry["type"] = "binary" if name in BINARY_FEATURES else "category"
            entry["allowed_values"] = {str(code): label for code, label in LABELS[name].items()}
        else:
            entry.update(type="number", min=BMI_RANGE[0], max=BMI_RANGE[1], unit="kg/m2")
        features.append(entry)
    return {
        "features": features,
        "output": {
            "risk_class": {str(code): label for code, label in RISK_LABELS.items()},
            "meaning": RISK_MEANING,
            "is_diagnosis": False,
            "probability_available": False,
        },
        "training_data": "CDC BRFSS 2015 health survey, self-reported answers from US adults",
    }


def describe_answers(features: dict) -> dict:
    """Turn answer codes into readable text, e.g. {'General health': 'Fair'}."""
    described = {}
    for name in FEATURES:
        value = features.get(name)
        if value is None:
            continue
        if name == "BMI":
            text = f"{float(value):.1f} kg/m2"
        else:
            text = LABELS[name].get(int(value), str(value))
        described[SHORT_NAMES[name]] = text
    return described

import logging
import os
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Diabetes Risk Check (educational demo)", layout="centered")

# src/ holds flat modules (not a package), so make it importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from inference import InputValidationError, PredictionError, predict_risk_class  # noqa: E402

logger = logging.getLogger(__name__)

# Optional, off by default: set PATIENT_LOG_API_URL (e.g. http://127.0.0.1:8000/predict)
# to send each submission to the local FastAPI/Postgres logger. Do NOT set this on a
# public deployment - the public app is designed not to store anyone's answers.
PATIENT_LOG_API_URL = os.environ.get("PATIENT_LOG_API_URL", "").strip()

# Answer labels follow the CDC BRFSS 2015 codebook; the numeric codes are exactly the
# values the model was trained on.
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

st.title("Diabetes Risk Check")
st.caption("An educational demo built on a public US health survey")

st.warning(
    "**For educational and informational purposes only.** This tool is **not** a medical "
    "diagnosis and does not replace professional medical advice, diagnosis or treatment. "
    "If you have any concerns about your health, please talk to a qualified healthcare professional."
)

if PATIENT_LOG_API_URL:
    st.info("Local test mode: your answers are sent to the configured patient-logging API.")
else:
    st.caption("Your answers are used only to produce the result below. This app does not save them.")

with st.form("risk_form"):
    st.subheader("Your answers")
    left, right = st.columns(2)

    high_bp = left.radio(
        "Have you been told by a doctor, nurse or other health professional that you have high blood pressure?",
        options=list(YES_NO), format_func=YES_NO.get, index=None, horizontal=True,
    )
    high_chol = right.radio(
        "Have you been told by a doctor, nurse or other health professional that your blood cholesterol is high?",
        options=list(YES_NO), format_func=YES_NO.get, index=None, horizontal=True,
    )
    bmi = left.number_input(
        "Body Mass Index (BMI), in kg/m²",
        min_value=12.0, max_value=98.0, value=None, step=0.1, format="%.1f", placeholder="e.g. 24.5",
        help="BMI = weight in kilograms divided by height in metres squared. Accepted range: 12 to 98.",
    )
    general_health = right.selectbox(
        "In general, would you say your health is:",
        options=list(GENERAL_HEALTH), format_func=GENERAL_HEALTH.get, index=None,
        placeholder="Select an option",
    )
    age_group = left.selectbox(
        "Age group (years)",
        options=list(AGE_GROUPS), format_func=AGE_GROUPS.get, index=None,
        placeholder="Select an option",
    )
    education = right.selectbox(
        "Highest grade or year of school completed",
        options=list(EDUCATION), format_func=EDUCATION.get, index=None,
        placeholder="Select an option",
    )
    income = st.selectbox(
        "Annual household income from all sources (US dollars)",
        options=list(INCOME), format_func=INCOME.get, index=None,
        placeholder="Select an option",
    )
    submitted = st.form_submit_button("Show result", type="primary")


def _send_to_logging_api(features):
    """Optional local-only logging hook. Never shows or logs exception details."""
    try:
        import requests

        requests.post(
            PATIENT_LOG_API_URL,
            json={name.lower(): float(value) for name, value in features.items()},
            timeout=5,
        ).raise_for_status()
    except Exception:
        st.caption("The optional logging service could not be reached; your result is unaffected.")


if submitted:
    answers = {
        "HighBP": high_bp, "HighChol": high_chol, "BMI": bmi, "GenHlth": general_health,
        "Age": age_group, "Income": income, "Education": education,
    }
    unanswered = [
        label for label, key in [
            ("High blood pressure", "HighBP"), ("High cholesterol", "HighChol"), ("BMI", "BMI"),
            ("General health", "GenHlth"), ("Age group", "Age"), ("Education", "Education"),
            ("Household income", "Income"),
        ] if answers[key] is None
    ]
    if unanswered:
        st.error("Please answer every question to see a result. Missing: " + ", ".join(unanswered) + ".")
    else:
        try:
            risk_class = predict_risk_class(answers)
        except InputValidationError as exc:
            st.error("Some answers are not valid:\n\n" + "\n".join(f"- {p}" for p in exc.problems))
        except PredictionError as exc:
            st.error(f"{exc} Please try again later.")
        except Exception:
            logger.exception("Unexpected error while producing a result")
            st.error("Something went wrong while producing your result. Please try again later.")
        else:
            st.subheader("Your result")
            if risk_class == 1:
                st.warning(
                    "**Higher-risk pattern.** Your answers are similar to those of survey respondents "
                    "who reported being told they have prediabetes or diabetes. This is **not** a "
                    "diagnosis, and many people with a similar pattern do not have diabetes. Consider "
                    "talking to a healthcare professional, who can arrange proper testing."
                )
            else:
                st.success(
                    "**Lower-risk pattern.** Your answers are more similar to those of survey respondents "
                    "who did not report prediabetes or diabetes. This does **not** rule out diabetes - "
                    "the tool misses some people who do have it. If you have any concerns, talk to a "
                    "healthcare professional."
                )
            st.info(
                "This is an educational demo, not medical advice. Only a clinical test ordered by a "
                "healthcare professional can diagnose diabetes."
            )
            if PATIENT_LOG_API_URL:
                _send_to_logging_api(answers)

with st.expander("How to read this result, and its limits"):
    st.markdown(
        """
- **What it compares.** The model compares your seven answers with about 254,000 responses to the
  2015 CDC Behavioral Risk Factor Surveillance System (BRFSS) health survey. "Higher risk" means your
  answers resemble those of respondents who reported prediabetes or diabetes.
- **Why no percentage.** The model is tuned to catch most people who report diabetes, so its raw scores
  overstate risk and are not a probability that you have diabetes. For that reason none is shown.
- **It makes mistakes in both directions.** Because it is tuned to catch most cases, it also flags many
  people who do not have diabetes, and it still misses some who do.
- **Only seven answers.** Family history, diet, physical activity, symptoms and blood tests are not
  included, and all of them matter.
- **Survey data has limits.** The responses are self-reported, come from US adults in 2015, and may not
  represent everyone equally.
"""
    )

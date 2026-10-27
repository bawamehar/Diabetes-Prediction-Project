import logging
import os
import re
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Diabetes Risk Check (educational demo)", layout="centered")

# Ensure src is importable (preprocess and dataload live under src/)
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

import diet_agent
from feature_schema import AGE_GROUPS, EDUCATION, GENERAL_HEALTH, INCOME, YES_NO
from inference import InputValidationError, PredictionError, predict_risk_class

logger = logging.getLogger(__name__)

# optional local logging, leave this unset on a public deployment
PATIENT_LOG_API_URL = os.environ.get("PATIENT_LOG_API_URL", "").strip()
MAX_PLANS_PER_SESSION = int(os.environ.get("DIET_AGENT_MAX_PER_SESSION", "3"))
# income and education are never sent to the AI
SHARED_FEATURES = ["HighBP", "HighChol", "BMI", "GenHlth", "Age"]

st.title("Diabetes Risk Check")
st.caption("An educational demo built on a public US health survey")

if PATIENT_LOG_API_URL:
    st.info("Local test mode: your answers are sent to the configured patient-logging API.")

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
    try:
        import requests

        requests.post(
            PATIENT_LOG_API_URL,
            json={name.lower(): float(value) for name, value in features.items()},
            timeout=5,
        ).raise_for_status()
    except Exception:
        st.caption("The optional logging service could not be reached; your result is unaffected.")


def show_result(risk_class):
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
            "who did not report prediabetes or diabetes. This does **not** rule out diabetes."
        )


def _plain(text):
    # the plan text comes from an LLM, show it as plain text rather than markdown
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~$])", r"\\\1", str(text))


def show_plan(plan):
    with st.container(border=True):
        st.markdown("#### General dietary guidance (AI-generated)")
        st.warning(
            "This was written by an AI model. It is general information, not individualized medical "
            "care, and it does not treat, cure or reverse diabetes. Do not change any medication or "
            "treatment because of it. Please talk to a qualified healthcare professional or registered "
            "dietitian about what is right for you."
        )
        st.markdown(_plain(plan.summary))
        for meal in plan.meals:
            st.markdown(f"**{_plain(meal.meal)}**")
            for idea in meal.ideas:
                st.markdown(f"- {_plain(idea)}")
        emphasize, limit = st.columns(2)
        emphasize.markdown("**Foods to emphasize**")
        for item in plan.foods_to_emphasize:
            emphasize.markdown(f"- {_plain(item)}")
        limit.markdown("**Foods to limit**")
        for item in plan.foods_to_limit:
            limit.markdown(f"- {_plain(item)}")
        st.markdown("**Practical tips**")
        for tip in plan.practical_tips:
            st.markdown(f"- {_plain(tip)}")


def show_diet_section(result):
    st.divider()
    st.subheader("Optional: AI dietary guidance")
    st.caption("General healthy-eating ideas written by an AI model. This is separate from the risk check above.")

    problem = diet_agent.config_problem()
    if problem:
        st.info(problem)
        return
    diet_agent.start_warm_up()

    with st.form("diet_form"):
        eating_pattern = st.selectbox("Eating pattern", diet_agent.EATING_PATTERNS)
        foods_to_avoid = st.text_input("Allergies or foods to avoid (optional)", max_chars=200)
        foods_liked = st.text_input("Foods or cuisines you enjoy (optional)", max_chars=200)
        left, right = st.columns(2)
        meals_per_day = left.number_input("Meals per day", min_value=2, max_value=6, value=3, step=1)
        budget = right.selectbox("Food budget", diet_agent.BUDGETS, index=1)
        cooking_time = st.selectbox("Time available to cook", diet_agent.COOKING_TIMES, index=1)
        share = st.checkbox(
            "Also share my risk-check result and answers (blood pressure, cholesterol, BMI, general health, "
            "age group) so the plan can be tailored."
        )
        st.caption(
            f"Your preferences (and your answers if ticked) are sent to the AI provider "
            f"({diet_agent.provider_name()}) to write the plan."
        )
        generate = st.form_submit_button("Generate dietary guidance")

    if generate:
        if st.session_state.get("plans_made", 0) >= MAX_PLANS_PER_SESSION:
            st.warning("You have reached the limit of AI plans for this session.")
        else:
            st.session_state["plans_made"] = st.session_state.get("plans_made", 0) + 1
            preferences = diet_agent.DietPreferences(
                eating_pattern=eating_pattern, foods_to_avoid=foods_to_avoid, foods_liked=foods_liked,
                meals_per_day=meals_per_day, budget=budget, cooking_time=cooking_time,
            )
            options = {}
            if share:
                options = {
                    "risk_class": result["risk_class"],
                    "shared_answers": {name: result["answers"][name] for name in SHARED_FEATURES},
                }
            try:
                with st.spinner("Writing your plan, this can take up to a minute..."):
                    st.session_state["diet_plan"] = diet_agent.generate_diet_plan(preferences, **options)
            except diet_agent.DietAgentUnavailable as exc:
                st.info(str(exc))
            except diet_agent.DietPlanError as exc:
                st.error(str(exc))
            except Exception:
                logger.exception("Unexpected error while writing the dietary plan")
                st.error("Something went wrong while writing the plan. Please try again later.")

    if "diet_plan" in st.session_state:
        show_plan(st.session_state["diet_plan"])


if submitted:
    st.session_state.pop("result", None)
    st.session_state.pop("diet_plan", None)
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
            st.session_state["result"] = {"answers": answers, "risk_class": risk_class}
            if PATIENT_LOG_API_URL:
                _send_to_logging_api(answers)

if "result" in st.session_state:
    show_result(st.session_state["result"]["risk_class"])
    show_diet_section(st.session_state["result"])

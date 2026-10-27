import importlib.util
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import diet_agent
import inference

UI_FILE = str(Path(__file__).resolve().parent.parent / "Streamlit" / "ui.py")


def new_app():
    return AppTest.from_file(UI_FILE, default_timeout=60).run()


def pick(elements, text):
    return next(element for element in elements if text in element.label)


def fill(at, *, high_bp, high_chol, bmi, general_health, age, education, income):
    pick(at.radio, "high blood pressure").set_value(high_bp)
    pick(at.radio, "cholesterol is high").set_value(high_chol)
    at.number_input[0].set_value(bmi)
    pick(at.selectbox, "your health is").select(general_health)
    pick(at.selectbox, "Age group").select(age)
    pick(at.selectbox, "school completed").select(education)
    pick(at.selectbox, "household income").select(income)


def submit_high_risk(at):
    fill(at, high_bp=1, high_chol=1, bmi=45.0, general_health=5, age=13, education=1, income=1)
    return at.button[0].click().run()


def submit_low_risk(at):
    fill(at, high_bp=0, high_chol=0, bmi=22.0, general_health=1, age=3, education=6, income=8)
    return at.button[0].click().run()


def alert_text(at):
    elements = list(at.warning) + list(at.success) + list(at.info) + list(at.error)
    return " ".join(element.value for element in elements)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("PATIENT_LOG_API_URL", "DIET_AGENT_MODEL", "DIET_AGENT_API_KEY", "OPENAI_API_KEY",
                 "DIET_AGENT_MAX_PER_SESSION"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def diet_configured(monkeypatch):
    pytest.importorskip("crewai")
    pytest.importorskip("mcp")
    monkeypatch.setenv("DIET_AGENT_MODEL", "openai/test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_page_loads_with_the_educational_line_and_no_result():
    at = new_app()
    assert not at.exception
    assert any("An educational demo built on a public US health survey" in caption.value for caption in at.caption)
    assert "Your result" not in [subheader.value for subheader in at.subheader]


def test_repeated_disclaimer_sections_are_gone():
    at = submit_high_risk(new_app())
    assert not at.exception
    assert not at.expander
    everything = page_text(at) + " " + " ".join(caption.value for caption in at.caption)
    for removed in (
        "For educational and informational purposes only",
        "This app does not save them",
        "This is an educational demo, not medical advice",
        "How to read this result",
    ):
        assert removed not in everything


def test_unanswered_questions_give_a_friendly_error():
    at = new_app()
    at.button[0].click().run()
    assert not at.exception
    assert "Please answer every question" in alert_text(at)
    assert "Your result" not in [subheader.value for subheader in at.subheader]


def test_out_of_range_bmi_is_treated_as_unanswered():
    # Streamlit drops numbers outside min/max, so the app sees BMI as missing
    at = new_app()
    fill(at, high_bp=1, high_chol=1, bmi=200.0, general_health=5, age=13, education=1, income=1)
    at.button[0].click().run()
    assert not at.exception
    text = alert_text(at)
    assert "Missing: BMI" in text
    assert "Higher-risk" not in text and "Lower-risk" not in text


def test_high_risk_answers_show_the_higher_risk_message():
    at = submit_high_risk(new_app())
    assert not at.exception
    assert "Higher-risk pattern" in alert_text(at)
    assert "not** a diagnosis" in alert_text(at)


def test_low_risk_answers_do_not_claim_diabetes_is_ruled_out():
    at = submit_low_risk(new_app())
    assert not at.exception
    text = alert_text(at)
    assert "Lower-risk pattern" in text
    assert "does **not** rule out diabetes" in text


def test_no_probability_or_percentage_is_shown():
    at = submit_high_risk(new_app())
    assert "%" not in alert_text(at)


def test_prediction_errors_are_shown_without_internal_details(monkeypatch):
    def broken(_features):
        raise RuntimeError("boom /internal/path secret-detail")

    monkeypatch.setattr(inference, "predict_risk_class", broken)
    at = submit_high_risk(new_app())
    assert not at.exception
    text = alert_text(at)
    assert "Something went wrong" in text
    assert "boom" not in text and "secret-detail" not in text


def test_model_unavailable_is_reported_politely(monkeypatch):
    def unavailable(_features):
        raise inference.ModelUnavailableError("The prediction model is currently unavailable.")

    monkeypatch.setattr(inference, "predict_risk_class", unavailable)
    at = submit_low_risk(new_app())
    assert not at.exception
    assert "currently unavailable" in alert_text(at)


def test_submissions_are_not_sent_anywhere_by_default(monkeypatch):
    import requests

    def fail(*_args, **_kwargs):
        raise AssertionError("the public app must not send submissions to any service")

    monkeypatch.setattr(requests, "post", fail)
    at = submit_high_risk(new_app())
    assert not at.exception
    assert "Higher-risk pattern" in alert_text(at)


def test_optional_logging_hook_is_opt_in_and_fails_quietly(monkeypatch):
    import requests

    calls = []

    def unreachable(url, json, timeout):
        calls.append((url, json))
        raise requests.ConnectionError("connection refused")

    monkeypatch.setenv("PATIENT_LOG_API_URL", "http://127.0.0.1:9/predict")
    monkeypatch.setattr(requests, "post", unreachable)
    at = submit_high_risk(new_app())
    assert not at.exception
    assert "Higher-risk pattern" in alert_text(at)
    assert len(calls) == 1
    assert set(calls[0][1]) == {"bmi", "age", "genhlth", "income", "highbp", "highchol", "education"}
    assert any("could not be reached" in caption.value for caption in at.caption)


PLAN = diet_agent.DietPlan(
    summary="A balanced day of vegetables and whole grains.",
    meals=[diet_agent.MealIdeas(meal="Breakfast", ideas=["Oats with berries"])],
    foods_to_emphasize=["Beans"],
    foods_to_limit=["Sugary drinks"],
    practical_tips=["Cook in batches"],
)


def page_text(at):
    return " ".join(element.value for element in at.markdown) + " " + alert_text(at)


def click(at, label):
    return pick(at.button, label).click().run()


def patch_generator(monkeypatch, result=PLAN):
    calls = []

    def fake(preferences, **options):
        calls.append((preferences, options))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(diet_agent, "generate_diet_plan", fake)
    return calls


def test_dietary_guidance_explains_when_it_is_not_set_up():
    at = submit_high_risk(new_app())
    assert not at.exception
    assert "not set up on this server" in alert_text(at)
    assert not at.text_input


def test_crewai_is_only_warmed_up_when_the_ai_form_is_shown(diet_configured, monkeypatch):
    started = []
    monkeypatch.setattr(diet_agent, "start_warm_up", lambda: started.append(1))
    assert not started
    submit_high_risk(new_app())
    assert started


def test_crewai_is_not_warmed_up_when_guidance_is_not_set_up(monkeypatch):
    started = []
    monkeypatch.setattr(diet_agent, "start_warm_up", lambda: started.append(1))
    submit_high_risk(new_app())
    assert not started


def test_dietary_form_says_which_answers_can_be_shared(diet_configured):
    at = submit_high_risk(new_app())
    assert not at.exception
    assert at.text_input
    label = at.checkbox[0].label
    assert "blood pressure, cholesterol, BMI, general health, age group" in label
    assert "income" not in label.lower() and "education" not in label.lower()


def test_plan_without_sharing_sends_no_health_answers(diet_configured, monkeypatch):
    calls = patch_generator(monkeypatch)
    at = click(submit_high_risk(new_app()), "Generate dietary guidance")
    assert not at.exception
    assert len(calls) == 1 and calls[0][1] == {}
    text = page_text(at)
    assert "General dietary guidance" in text
    assert "Oats with berries" in text


def test_the_ai_notice_and_provider_caption_are_not_repeated(diet_configured, monkeypatch):
    patch_generator(monkeypatch)
    at = click(submit_high_risk(new_app()), "Generate dietary guidance")
    assert not at.exception
    everything = page_text(at) + " " + " ".join(caption.value for caption in at.caption)
    assert "General dietary guidance" in everything
    assert "This was written by an AI model" not in everything
    assert "sent to the AI provider" not in everything


def test_plan_is_shown_separately_from_the_prediction(diet_configured, monkeypatch):
    patch_generator(monkeypatch)
    at = click(submit_high_risk(new_app()), "Generate dietary guidance")
    subheaders = [subheader.value for subheader in at.subheader]
    assert subheaders.index("Your result") < subheaders.index("AI dietary guidance")
    assert "Higher-risk pattern" in alert_text(at)


def test_sharing_sends_only_the_needed_answers(diet_configured, monkeypatch):
    calls = patch_generator(monkeypatch)
    at = submit_high_risk(new_app())
    at.checkbox[0].check()
    at = click(at, "Generate dietary guidance")
    assert not at.exception
    options = calls[0][1]
    assert options["risk_class"] == 1
    assert set(options["shared_answers"]) == {"HighBP", "HighChol", "BMI", "GenHlth", "Age"}


def test_plan_text_is_shown_as_plain_text(diet_configured, monkeypatch):
    nasty = diet_agent.DietPlan(
        summary="**bold** ![x](http://example.com/a.png)",
        meals=[diet_agent.MealIdeas(meal="Lunch", ideas=["[link](http://example.com)"])],
        foods_to_emphasize=[], foods_to_limit=[], practical_tips=[],
    )
    patch_generator(monkeypatch, nasty)
    at = click(submit_high_risk(new_app()), "Generate dietary guidance")
    text = page_text(at)
    assert r"\*\*bold\*\*" in text and r"\[link\]" in text


def test_plan_failures_are_shown_without_internal_details(diet_configured, monkeypatch):
    patch_generator(monkeypatch, diet_agent.DietPlanError("The generated plan was incomplete. Please try again."))
    at = click(submit_high_risk(new_app()), "Generate dietary guidance")
    assert not at.exception
    assert "incomplete" in alert_text(at)

    patch_generator(monkeypatch, RuntimeError("boom /internal/path secret-detail"))
    at = click(submit_high_risk(new_app()), "Generate dietary guidance")
    assert not at.exception
    assert "Something went wrong" in alert_text(at)
    assert "boom" not in alert_text(at) and "secret-detail" not in alert_text(at)


def test_unavailable_dietary_guidance_does_not_break_the_page(diet_configured, monkeypatch):
    patch_generator(monkeypatch, diet_agent.DietAgentUnavailable("AI dietary guidance could not be started on this server."))
    at = click(submit_high_risk(new_app()), "Generate dietary guidance")
    assert not at.exception
    assert "could not be started" in alert_text(at)
    assert "Higher-risk pattern" in alert_text(at)


def test_plans_are_limited_per_session(diet_configured, monkeypatch):
    monkeypatch.setenv("DIET_AGENT_MAX_PER_SESSION", "1")
    calls = patch_generator(monkeypatch)
    at = click(submit_high_risk(new_app()), "Generate dietary guidance")
    at = click(at, "Generate dietary guidance")
    assert len(calls) == 1
    assert "limit of AI plans" in alert_text(at)


def test_plan_is_cleared_when_answers_are_resubmitted(diet_configured, monkeypatch):
    patch_generator(monkeypatch)
    at = click(submit_high_risk(new_app()), "Generate dietary guidance")
    assert "diet_plan" in at.session_state
    at = submit_low_risk(at)
    assert "diet_plan" not in at.session_state
    assert "Lower-risk pattern" in alert_text(at)


def test_page_still_works_when_crewai_is_not_installed(monkeypatch):
    monkeypatch.setenv("DIET_AGENT_MODEL", "openai/test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *args: None if name == "crewai" else real_find_spec(name, *args)
    )
    at = submit_high_risk(new_app())
    assert not at.exception
    assert "Higher-risk pattern" in alert_text(at)
    assert "not installed on this server" in alert_text(at)
    assert not at.text_input

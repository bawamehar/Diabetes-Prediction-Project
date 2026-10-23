from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

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
def logging_disabled(monkeypatch):
    monkeypatch.delenv("PATIENT_LOG_API_URL", raising=False)


def test_page_loads_with_disclaimer_and_no_result():
    at = new_app()
    assert not at.exception
    text = alert_text(at)
    assert "educational and informational purposes only" in text
    assert "not** a medical diagnosis" in text
    assert "Your result" not in [subheader.value for subheader in at.subheader]


def test_unanswered_questions_give_a_friendly_error():
    at = new_app()
    at.button[0].click().run()
    assert not at.exception
    assert "Please answer every question" in alert_text(at)
    assert "Your result" not in [subheader.value for subheader in at.subheader]


def test_out_of_range_bmi_is_treated_as_unanswered():
    # Streamlit drops a number outside the widget's min/max (also when sent by a
    # hand-crafted client), so the app reports it as missing instead of predicting.
    # Range checks inside predict_risk_class are covered in test_inference.py.
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
    # The model is trained with class-balanced weights, so its scores are not real probabilities.
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

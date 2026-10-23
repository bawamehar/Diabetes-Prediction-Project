# Diabetes Risk Check

![Diabetes Prediction Project](diabetes-pred.png)

An educational web app that compares a person's answers to seven simple questions with patterns in a
large public US health survey, and reports whether those answers look more like respondents who
reported prediabetes or diabetes ("higher-risk pattern") or not ("lower-risk pattern").

> **For educational and informational purposes only.** This project is not a medical device, does not
> diagnose diabetes, and is not a substitute for professional medical advice, diagnosis or treatment.
> Anyone with health concerns should talk to a qualified healthcare professional.

## What the app asks

The model uses exactly seven inputs. The numeric codes are what the model was trained on; the app shows
the matching plain-language answers (labels follow the CDC BRFSS 2015 codebook).

| Feature | Question in the app | Accepted values |
|---|---|---|
| `HighBP` | Told by a health professional you have high blood pressure? | 0 = No, 1 = Yes |
| `HighChol` | Told by a health professional your blood cholesterol is high? | 0 = No, 1 = Yes |
| `BMI` | Body Mass Index, kg/m² | 12 to 98 |
| `GenHlth` | General health | 1 Excellent, 2 Very good, 3 Good, 4 Fair, 5 Poor |
| `Age` | Age group | 1 = 18-24, 2 = 25-29, ... 13 = 80 or older (5-year groups) |
| `Education` | Highest school grade completed | 1 to 6 (1 = never attended / kindergarten only, 6 = college graduate) |
| `Income` | Annual household income (USD) | 1 to 8 (1 = under $10,000, 8 = $75,000 or more) |

Every question is required. Values outside these ranges are rejected with a clear message (see
`src/inference.py`).

## How the model works

- **Data:** about 254,000 responses to the 2015 CDC Behavioral Risk Factor Surveillance System (BRFSS)
  health survey (`Data/data/diabetes.csv`). Answers are self-reported.
- **Target:** class 1 means the respondent reported **prediabetes or diabetes**; class 0 means neither.
  About 16% of respondents are class 1.
- **Model:** a scikit-learn `Pipeline` (`StandardScaler` + `GradientBoostingClassifier`) saved at
  `models/best_model.joblib`. `src/train.py` compares seven candidates (logistic regression, decision
  tree, random forest, gradient boosting, AdaBoost, XGBoost, LightGBM) with 5-fold cross-validation and
  keeps the best by a weighted score: 40% accuracy, 30% recall, 30% ROC AUC.
- **Class balancing:** because only ~16% of respondents are class 1, training weights the minority class
  up. Without this every model collapses to answering "no" almost every time. The price is that the
  model is deliberately sensitive: it flags many people who do not have diabetes.
- **Measured on 25,368 held-out survey responses** (never used to fit this model):

  | Accuracy | Recall | Precision | ROC AUC | Flagged "higher risk" |
  |---|---|---|---|---|
  | 71.9% | 78.9% | 33.4% | 0.825 | 37.2% of people |

  In plain terms: it catches about 8 in 10 respondents who reported prediabetes/diabetes, misses about
  2 in 10, and only about 1 in 3 of the people it flags actually reported it. Re-running `train.py`
  uses a fresh 90/10 split, so those figures will shift slightly.
- **No percentage is shown, on purpose.** Because of the class balancing, the model's raw scores are not
  probabilities. On the held-out data the average score was 0.39 while the true rate was 0.16, and among
  people scored 0.8-0.9 only about 52% actually reported diabetes. The app therefore shows only the
  higher-/lower-risk pattern, never a "chance of diabetes".

## Run it locally

Requires Python 3.12 (see `.python-version`). Commands below are for Windows PowerShell; on macOS/Linux
use `source .venv/bin/activate` instead of the activate line.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run Streamlit/ui.py
```

Open <http://localhost:8501>. `requirements.txt` holds only what the app needs, pinned to the versions
the model was trained with (loading a scikit-learn pickle under a different version can warn or fail).
For development (tests, training, optional API) install `requirements-dev.txt` instead.

### Try these inputs

| | HighBP | HighChol | BMI | General health | Age group | Education | Income | Expected result |
|---|---|---|---|---|---|---|---|---|
| A | Yes | Yes | 45 | Poor | 80 or older | Never attended school | Under $10,000 | Higher-risk pattern |
| B | No | No | 22 | Excellent | 30 to 34 | College graduate | $75,000 or more | Lower-risk pattern |
| C | Yes | No | 32 | Good | 60 to 64 | High school graduate | $25,000 to $35,000 | Higher-risk pattern |

Submitting with any question unanswered shows "Please answer every question" and lists what is missing.

### Tests

```powershell
pip install -r requirements-dev.txt
pytest
```

The suite covers model loading, input validation (missing, malformed and out-of-range values), the
Streamlit page (both result messages, no percentage shown, friendly errors, no data sent anywhere by
default), and the optional API. API tests skip themselves if FastAPI is not installed.

## Retraining the model (manual)

Retraining is a deliberate, manual step; nothing retrains automatically.

```powershell
pip install -r requirements-dev.txt
python src/train.py
```

This overwrites `models/best_model.joblib` and prints test-set accuracy, recall, precision and ROC AUC.
Review those numbers before committing a new model, and keep the pinned scikit-learn version in
`requirements.txt` in step with the version that produced the file.

## Optional: patient-logging API (local only, off by default)

`api/` contains a small FastAPI service that can store submitted answers in a local PostgreSQL table.
**The public app does not use it** and does not store anyone's answers. To try it locally:

```powershell
$env:DB_HOST = "localhost"; $env:DB_NAME = "postgres"; $env:DB_USER = "postgres"
$env:DB_PASSWORD = "<your password>"     # never commit real credentials
uvicorn api.fastapi:app --port 8000
# in another terminal, opt the app in:
$env:PATIENT_LOG_API_URL = "http://127.0.0.1:8000/predict"
streamlit run Streamlit/ui.py
```

## Deploying publicly

See [DEPLOYMENT.md](DEPLOYMENT.md) for the hosting comparison, privacy considerations, GitHub steps and a
step-by-step deployment guide.

## Project layout

```
Streamlit/ui.py        the public app (questions, validation messages, result)
src/inference.py       model loading, input validation, prediction
src/preprocess.py      feature list and input alignment
src/train.py           trains/compares candidate models and saves the best
src/evaluate.py        cross-validation and the weighted model-selection score
src/predict.py         command-line trainer/predictor (older script)
models/best_model.joblib   the trained scaler + classifier pipeline
Data/data/diabetes.csv     training data (BRFSS 2015)
api/                   optional local patient-logging API (not used by the public app)
tests/                 pytest suite
.streamlit/config.toml hides tracebacks, disables usage telemetry
```

## Privacy

The public app holds answers in memory only for your browser session while it computes the result. It
does not write them to a database or file and does not log them. Hosting platforms may still keep ordinary web-server
metadata (such as IP addresses and request times) under their own policies. Review
[DEPLOYMENT.md](DEPLOYMENT.md#privacy-security-and-regulatory-notes) before opening the app to the public.

## Limitations

- Trained on 2015 self-reported US survey data; it may not represent everyone equally.
- Only seven inputs. Family history, diet, activity, symptoms and lab tests are not considered.
- Tuned to catch most people who report diabetes, so it produces many false alarms and still misses some.
- It shows a screening-style pattern, not a diagnosis. Only a clinical test can diagnose diabetes.

## Data source

CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey data, as distributed in the public
"Diabetes Health Indicators" dataset on Kaggle / UCI. Check the dataset's current license and
acknowledgement requirements before redistributing it.

# Diabetes Risk Check

![Diabetes Prediction Project](diabetes-pred.png)

An educational web app that compares a person's answers to seven simple questions with patterns in a
large public US health survey, and reports whether those answers look more like respondents who
reported prediabetes or diabetes ("higher-risk pattern") or not ("lower-risk pattern").

Link: [Diabetes Risk Prediction & Diet Plan](https://patient-diabetes-prediction.streamlit.app/)

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
  2 in 10, and only about 1 in 3 of the people it flags actually reported it.
- **Where this exact file came from:** it was trained on 190,265 of the 228,312 non-held-out rows, by the
  earlier retraining workflow that has since been removed. `src/train.py` trains on all 228,312, so
  running it produces a different model (same held-out rows, so the numbers stay comparable).
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
default), the MCP server (schemas, validation, a real stdio session), the CrewAI agent and the dietary
section of the page, the model card check, MLflow logging and the registry (in a temporary store), and
the optional API. API and MLflow tests skip themselves if FastAPI or MLflow is not installed.
The LLM is always replaced by a scripted stand-in or a local mock of the provider's HTTP API, so no key
or network is needed and nothing here proves a real model answers well.

## Retraining the model (manual)

Retraining is a deliberate, manual step; nothing retrains automatically.

```powershell
pip install -r requirements-dev.txt
python src/train.py
```

This overwrites `models/best_model.joblib`, prints test-set accuracy, recall, precision and ROC AUC, and
records the run in MLflow (next section). Review those numbers before committing a new model, and keep the
pinned scikit-learn version in `requirements.txt` in step with the version that produced the file.

## Experiment tracking and model registry (MLflow)

MLflow is for development only: it is in `requirements-dev.txt` and the deployed app never imports it.

- **Tracking.** Each `src/train.py` run is logged to the experiment `diabetes-risk`: algorithm,
  hyperparameters, split settings, row counts, a hash of the data file, the holdout metrics (accuracy,
  precision, recall, F1, ROC AUC, confusion counts), the cross-validation scores of the chosen model, the
  candidate ranking, the feature order and the model itself. No survey rows or answers are logged.
- **Registry.** The model is registered as a new version of `diabetes-risk-classifier`. Registering does
  **not** change what the app serves.
- **Choosing the served version.**

  ```powershell
  python src/registry.py show          # versions, the champion alias, and whether models/ matches the card
  python src/registry.py promote 2     # serve version 2
  ```

  `promote` sets the `champion` alias, puts that version's original `.joblib` file at
  `models/best_model.joblib` and rewrites `models/model_card.json`. Commit both to deploy it.
- **Looking at it in the browser.** From the project folder:

  ```powershell
  .\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow_store/mlflow.db
  ```

  Use the project's own interpreter as shown (or activate the venv first). A bare `mlflow` command can
  start another install, for example Anaconda's older MLflow, which fails with
  `Can't locate revision identified by 'b7e2c1a4d9f3'` because the store was created by a newer version;
  that failure changes nothing in the store. Open <http://127.0.0.1:5000> (the first start can take a
  minute) and click **Model training**: *Experiments* lists the runs (parameters, metrics, artifacts) and
  *Model registry* lists `diabetes-risk-classifier` with its versions and the `champion` alias. Stop it
  with Ctrl+C. It only reads the local store.
- **How the app knows its version.** `models/model_card.json` names the registered model, version and
  SHA-256 of `models/best_model.joblib`. `src/inference.py` checks the file against it when it loads the
  model and also checks the feature order, so a file that was swapped by hand or by `train.py` is refused
  (the page says the model is unavailable) rather than served under the wrong version. The server log
  shows `Serving diabetes-risk-classifier version N` once at startup.
- **Where it is stored.** By default in `mlflow_store/` (a SQLite database plus artifacts, git-ignored). Set
  `MLFLOW_TRACKING_URI` to use a different database or a tracking server, and `MLFLOW_EXPERIMENT_NAME` to
  change the experiment. Nothing under `mlflow_store/` exists on a hosting platform, and files written there
  would not survive a restart, so the registry is a local record; the model card is what travels.
- **Version history.** Neither earlier model had an MLflow run, so `python src/backfill_registry.py --apply`
  (already run once on this machine; it refuses to run twice) registered them as runs that say so:
  version 1 is the original AdaBoost model recovered from git (commit `11ed81e`, 21 inputs, never
  evaluated here, cannot be served by this app) and version 2 is the current file, with its holdout
  metrics recomputed from the saved model. Versions are numbered by registration order, so that is the only
  way the current model could honestly be version 2. On a new machine, run the same command to rebuild the
  local registry, or ignore it if you only need the app.


## AI dietary guidance (CrewAI + MCP)

After a result, the app can ask an LLM for general meal suggestions. It is off until you configure a model,
it is shown separately from the risk check, and it is labelled as AI dietary guidance.

How it fits together:

- `src/diet_agent.py` runs one CrewAI agent with one task (one LLM call) that returns a structured plan.
- `src/mcp_server.py` is a local MCP server (stdio) with read-only tools. `get_feature_schema` explains the
  model inputs and what the output means. `get_diabetes_prediction` calls the existing `inference.py` and
  is only served with `--with-prediction`. The agent connects to it as a subprocess; nothing is exposed
  on the network.
- When the user shares their result, the agent is given it (plain words, no tool call needed) and only the
  schema tool. The prediction tool is offered only if the result is missing.
- `src/feature_schema.py` holds the answer labels and limits so the UI, the API and the MCP server agree.

Configuration (environment variables, or Streamlit secrets on Community Cloud):

| Variable | Purpose |
|---|---|
| `DIET_AGENT_MODEL` | Required. CrewAI model name, e.g. `openai/<model>`, `anthropic/<model>`, `gemini/<model>` |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` | The key for that provider (or set `DIET_AGENT_API_KEY`) |
| `DIET_AGENT_BASE_URL` | Optional, for an OpenAI-compatible server (for example a local model) |
| `DIET_AGENT_MAX_PER_SESSION` | Plans allowed per browser session, default 3 |
| `DIET_AGENT_TEMPERATURE`, `DIET_AGENT_MAX_TOKENS` | Optional. Left unset they are not sent, because the gpt-5 models reject them |
| `DIET_AGENT_REASONING_EFFORT` | Optional (`minimal`, `low`, `medium`, `high`), for reasoning models that accept it |

On your machine, set them in the shell:

```powershell
$env:DIET_AGENT_MODEL = "openai/<model name>"
$env:OPENAI_API_KEY = "<your key>"      # never commit keys
streamlit run Streamlit/ui.py
```

On Streamlit Community Cloud, paste this (TOML) into **Advanced settings -> Secrets**. The names must be at
the top level, above any `[section]` header: only top-level secrets become environment variables, and
this code reads environment variables.

```toml
DIET_AGENT_MODEL = "openai/<model name>"
OPENAI_API_KEY = "<your key>"
```

## Project layout

```
Streamlit/ui.py        the public app (questions, validation messages, result)
src/inference.py       model loading, input validation, prediction
src/feature_schema.py  feature limits, answer labels, output meaning (shared by app, API, MCP)
src/diet_agent.py      CrewAI dietary guidance agent (optional)
src/mcp_server.py      local MCP server exposing the schema and, optionally, the prediction
src/preprocess.py      feature list and input alignment
src/train.py           trains/compares candidate models, saves the best and logs it to MLflow
src/evaluate.py        cross-validation and the weighted model-selection score
src/predict.py         command-line trainer/predictor (older script)
src/model_card.py      checks models/best_model.joblib against models/model_card.json (no MLflow needed)
src/tracking.py        MLflow logging and registration helpers (development only)
src/registry.py        `show` / `promote` commands for choosing the served model version
src/backfill_registry.py   one-off registration of the two existing models
models/best_model.joblib   the trained scaler + classifier pipeline
models/model_card.json     which registered version that file is, with its SHA-256 and holdout metrics
Data/data/diabetes.csv     training data (BRFSS 2015)
api/                   optional local patient-logging API (not used by the public app)
tests/                 pytest suite
.streamlit/config.toml hides tracebacks, disables usage telemetry
```

## Data source

CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey data, as distributed in the public
"Diabetes Health Indicators" dataset on Kaggle / UCI. Check the dataset's current license and
acknowledgement requirements before redistributing it.

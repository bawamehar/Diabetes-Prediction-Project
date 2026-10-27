from pathlib import Path
import joblib
from evaluate import classification_cv, select_best_model, get_candidate_models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import model_card
import tracking
from preprocess import preprocess_data
from dataload import read_data, DATA_FILE
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

df = read_data()

independent_vars, dependent_var = preprocess_data(df)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    independent_vars, dependent_var, test_size=0.1, random_state=25, stratify=dependent_var
)

# classes are about 84/16, weight them or the models mostly predict "no diabetes"
sample_weight_train = compute_sample_weight('balanced', y_train)

BASE_DIR = Path(__file__).resolve().parent.parent
Model_loc = BASE_DIR / "models"
MODEL_FILENAME = "best_model.joblib"


models = get_candidate_models()


results = {}

for name, model in models.items():
    print(f"\nEvaluating {name}")
    # Evaluate on the training set using cross-validation
    df_metrics = classification_cv(model, X=X_train, y=y_train, sample_weight=sample_weight_train)
    results[name] = df_metrics


best_model, scores_df  = select_best_model(results)
model = models[best_model] # actucal model object


def train_best_model(best_model=model):
    # Fit best model on training data
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', best_model)])
    pipeline.fit(X_train, y_train, clf__sample_weight=sample_weight_train)
    # Evaluate on hold-out test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    # Save the trained model
    Model_loc.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, Model_loc / MODEL_FILENAME)
    print(f"Best model {best_model} trained and saved successfully. "
          f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}, "
          f"recall: {recall_score(y_test, y_pred):.4f}, "
          f"precision: {precision_score(y_test, y_pred):.4f}, "
          f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    return pipeline, tracking.holdout_metrics(y_test, y_pred, y_proba)

pipeline, holdout_metrics = train_best_model()


# Track the run and register it as a new model version. This never changes the version the app serves.
def percent(value):
    return float(value.strip('%')) / 100

try:
    cv_scores = results[best_model].iloc[0]
    run_id, version = tracking.log_run(
        pipeline,
        run_name=f"train {best_model}",
        params={
            "holdout_fraction": 0.1, "split_random_state": 25, "stratified": True,
            "class_weighting": "balanced", "cv_folds": 5,
            "training_rows": len(X_train), "holdout_rows": len(X_test),
        },
        metrics={
            **holdout_metrics,
            **{f"cv_{name}": percent(cv_scores[column]) for name, column in [
                ("train_accuracy", "Training Accuracy"), ("accuracy", "Test Accuracy"), ("precision", "Precision"),
                ("recall", "Recall"), ("f1", "F1-Score"), ("roc_auc", "ROC AUC")]},
        },
        tags={
            "source": "src/train.py",
            "selection": "weighted score of CV test accuracy, recall and ROC AUC (evaluate.select_best_model)",
            **tracking.dataset_tags(DATA_FILE, len(df)),
        },
        model_file=Model_loc / MODEL_FILENAME,
        version_tags={"source": "training_run"},
        description=f"{best_model} trained by src/train.py",
        extras={"candidate_ranking.json": scores_df.to_dict(orient="records")},
    )
    print(f"Registered {tracking.REGISTERED_MODEL} version {version} (MLflow run {run_id}).")
    card = model_card.read_card()
    if card and card["sha256"] != model_card.file_sha256(Model_loc / MODEL_FILENAME):
        print(f"{MODEL_FILENAME} now differs from models/model_card.json, so the app will not load it until you "
              f"choose a version: python src/registry.py promote {version}")
except tracking.MlflowNotInstalled as exc:
    print(f"Model saved but not tracked: {exc}")
except Exception as exc:
    print(f"Model saved but MLflow tracking failed: {type(exc).__name__}: {exc}")

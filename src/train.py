from pathlib import Path
import joblib
from evaluate import classification_cv, select_best_model, get_candidate_models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import preprocess_data
from dataload import read_data
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

df = read_data()

independent_vars, dependent_var = preprocess_data(df)

# Hold out 10% as a test set that is never used for model selection or fitting.
X_train, X_test, y_train, y_test = train_test_split(
    independent_vars, dependent_var, test_size=0.1, random_state=25, stratify=dependent_var
)

# The dataset is ~84% non-diabetic / ~16% diabetic. Without correcting for this,
# every model collapses toward always predicting the majority class (near-zero
# recall). sample_weight='balanced' reweights the minority class during fitting.
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
    # Bundle scaling + classifier into one pipeline so the exact scaling used at
    # training time travels with the saved artifact - avoids the model being served
    # with a scaler fit on different data than it was trained on.
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', best_model)])
    pipeline.fit(X_train, y_train, clf__sample_weight=sample_weight_train)
    # Evaluate on the hold-out test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    # Save the trained pipeline under a stable filename so downstream consumers
    # (Streamlit UI, tests) don't need to know which algorithm won
    Model_loc.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, Model_loc / MODEL_FILENAME)
    print(f"Best model {best_model} trained and saved successfully. "
          f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}, "
          f"recall: {recall_score(y_test, y_pred):.4f}, "
          f"precision: {precision_score(y_test, y_pred):.4f}, "
          f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

train_best_model()

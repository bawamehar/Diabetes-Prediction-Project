import joblib
import numpy as np
import pandas as pd
import os
from dataload import read_data
from preprocess import preprocess_data
from evaluate import classification_cv, select_best_model, get_candidate_models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_user_input(feature_names):
    """
    Prompt user for each feature value and return as a DataFrame.
    """
    input_data = {}
    for feature in feature_names:
        val = input(f"Enter value for {feature}: ")
        try:
            val = float(val)
        except ValueError:
            pass
        input_data[feature] = [val]
    return pd.DataFrame(input_data)


def main():
    # 1. Load and preprocess data
    df = read_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=25, stratify=y
    )

    # 2. Train all models and evaluate using classification_cv
    models = get_candidate_models()
    sample_weight = compute_sample_weight('balanced', y_train)

    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}")
        df_metrics = classification_cv(model, X_train, y_train, sample_weight=sample_weight)
        results[name] = df_metrics

    print(df.head(5))
    # 3. Select best model using weighted metric
    best_model_name, scores_df = select_best_model(results)
    best_model = models[best_model_name]
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', best_model)])
    pipeline.fit(X_train, y_train, clf__sample_weight=sample_weight)
    # Save the best model
    model_path = os.path.join(os.path.dirname(__file__), '../models/best_model.joblib')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Best model: {best_model_name} (saved to {model_path})")

    # 4. Get user input and predict
    feature_names = list(X.columns)

    print("Please enter the following features:")
    user_df = get_user_input(feature_names)
    prediction = pipeline.predict(user_df)
    print(f"Predicted Diabetes (0=No, 1=Yes): {prediction[0]}")

if __name__ == "__main__":
    main()

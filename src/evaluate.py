import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    make_scorer, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from preprocess import preprocess_data
from dataload import read_data

df = read_data()
independent_vars, dependent_var = preprocess_data(df)


def get_candidate_models():
    """Models compared in train.py and predict.py."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=52),
        'Random Forest': RandomForestClassifier(random_state=52),
        'gradient boost': GradientBoostingClassifier(random_state=52),
        'adaboost': AdaBoostClassifier(random_state=52),
        'XGBoost': XGBClassifier(eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=52, verbose=-1),
    }


def classification_cv(model, X=independent_vars, y=dependent_var, cv_splits=5, sample_weight=None):
    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': 'roc_auc'
    }

    # Create stratified K-Fold cross-validator
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # scaler sits inside the pipeline so every fold fits its own
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', model)])
    fit_params = {'clf__sample_weight': sample_weight} if sample_weight is not None else None

    # Run cross-validation
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True,
                                 params=fit_params)

    # Collect average metrics
    df_metrics = pd.DataFrame({
        'Training Accuracy': [cv_results['train_accuracy'].mean()],
        'Test Accuracy': [cv_results['test_accuracy'].mean()],
        'Precision': [cv_results['test_precision'].mean()],
        'Recall': [cv_results['test_recall'].mean()],
        'F1-Score': [cv_results['test_f1'].mean()],
        'ROC AUC': [cv_results['test_roc_auc'].mean()]
    })

    # Convert values to percentages
    for col in df_metrics.columns:
        df_metrics[col] = df_metrics[col].map(lambda x: f'{x:.2%}')
    
    # Print the DataFrame
    print(df_metrics)

    return df_metrics

# think about fuction whihc one to take then use that model  to dumbh and predict
def select_best_model(results, weight_accuracy=0.4, weight_recall=0.3, weight_rocauc=0.3):
    scores = []

    for model_name, df_metrics in results.items():
        # Convert percentage strings back to float
        test_acc = float(df_metrics['Test Accuracy'][0].strip('%')) / 100
        recall = float(df_metrics['Recall'][0].strip('%')) / 100
        roc_auc = float(df_metrics['ROC AUC'][0].strip('%')) / 100

        # Weighted score
        total_score = (
            (test_acc * weight_accuracy) +
            (recall * weight_recall) +
            (roc_auc * weight_rocauc)
        )

        scores.append({
            'Model': model_name,
            'Test Accuracy': test_acc,
            'Recall': recall,
            'ROC AUC': roc_auc,
            'Weighted Score': total_score
        })

    # Convert to DataFrame for sorting
    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.sort_values(by='Weighted Score', ascending=False).reset_index(drop=True)

    # Best model
    best_model = scores_df.iloc[0]

    print("\nModel Ranking:")
    print(scores_df)

    print(f"\nBest model based on weights: {best_model['Model']}")
    print(f"Test Accuracy: {best_model['Test Accuracy']:.2%}")
    print(f"Recall: {best_model['Recall']:.2%}")
    print(f"ROC AUC: {best_model['ROC AUC']:.2%}")

    return best_model['Model'], scores_df

# best_model = select_best_model(results)
# print(f"\nBest model to use for predictions: {best_model[0]}")
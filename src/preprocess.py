from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("preprocess.py loaded")

# Features collected by the Streamlit UI / API. The model is trained on exactly
# this set so there's no train/serve mismatch (previously the model was trained
# on all 21 columns and the UI's missing 14 were filled with dataset medians,
# which diluted the risk signal from the 7 features users actually enter).
UI_FEATURES = ['HighBP', 'HighChol', 'BMI', 'GenHlth', 'Age', 'Income', 'Education']

def preprocess_data(df):
    df['Diabetes_012'] =df['Diabetes_012'].replace({2:1})
    df = df.rename(columns = {'Diabetes_012': 'Diabetes_binary'})
    target = 'Diabetes_binary'
    independent_vars = df[UI_FEATURES]
    dependent_var = df[target]

    return independent_vars, dependent_var

def train_test_data(X, y, test_size=0.1, random_state=25):
   

    scaler = StandardScaler()

    #Fit only on training data
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def get_feature_names():
    """Return the list of independent variable names used for training."""
    return list(UI_FEATURES)


def transform_user_input(user_df):
    """
    Reorder/align a single-row (or multi-row) DataFrame with the UI feature columns
    to match the training column order. Scaling is handled internally by the saved
    model pipeline (StandardScaler + classifier), not here.
    """
    return user_df.reindex(columns=UI_FEATURES)


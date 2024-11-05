
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load dataset from the specified file path.
    """
    return pd.read_csv(file_path)

def handle_missing_values(df):
    """
    Handle missing values in the dataset by filling with median for numerical
    and mode for categorical features.
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    return df

def encode_categorical_features(df):
    """
    One-hot encode categorical features in the dataset.
    """
    return pd.get_dummies(df, drop_first=True)

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Standardize numerical features for model training.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

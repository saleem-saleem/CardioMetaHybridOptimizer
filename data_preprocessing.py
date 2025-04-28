import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    """Load CSV dataset and preprocess (scaling, NaN handling)."""
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df = df.dropna()
    
    # Separate features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

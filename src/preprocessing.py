import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath, feature_columns=None):
    """
    Load CSV file and preprocess features
    
    Args:
        filepath: Path to CSV file
        feature_columns: List of feature column names (if None, auto-detect)
    
    Returns:
        X_scaled: Normalized feature matrix
        df: Original dataframe
        scaler: Fitted StandardScaler object
        selected_features: List of feature column names
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Auto-detect feature columns if not provided
    if feature_columns is None:
        exclude_cols = ['filename', 'label', 'length', 'Unnamed: 0']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Selected {len(feature_columns)} features")
    
    # Extract features
    X = df[feature_columns].values
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("✓ Data preprocessing completed")
    
    return X_scaled, df, scaler, feature_columns

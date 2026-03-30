import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in the dataframe.
    
    Args:
        df (pd.DataFrame): Input pandas DataFrame.
        strategy (str): 'drop' to drop rows with missing values, 'fill' to fill them.
        
    Returns:
        pd.DataFrame: DataFrame with handled missing values.
    """
    df_clean = df.copy()
    
    if strategy == 'drop':
        df_clean = df_clean.dropna()
    elif strategy == 'fill':
        # Fill numerical columns with median
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        if not num_cols.empty:
            for col in num_cols:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                
        # Fill categorical columns with mode
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        if not cat_cols.empty:
            for col in cat_cols:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    else:
        raise ValueError("Strategy must be either 'drop' or 'fill'")
        
    return df_clean

def encode_categorical_features(df: pd.DataFrame, categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Encode categorical variables using pd.get_dummies.
    
    Args:
        df (pd.DataFrame): Input pandas DataFrame.
        categorical_columns (List[str], optional): Columns to encode. Auto-detects if None.
        
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded categorical variables.
    """
    # Auto-detect categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
    # Apply get_dummies (drop_first=True to avoid multicollinearity dummy trap)
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    return df_encoded

def separate_features_target(df: pd.DataFrame, target_column: str = 'math score') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate the dataset into features (X) and target variable (y).
    
    Args:
        df (pd.DataFrame): Input pandas DataFrame.
        target_column (str): Name of the target column.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: X (features dataframe), y (target series).
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataframe.")
        
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y

def run_preprocessing_pipeline(
    df: pd.DataFrame, 
    target_column: str = 'math score', 
    missing_strategy: str = 'drop'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Run the full preprocessing pipeline on the student exam dataset.
    
    Args:
        df (pd.DataFrame): Input pandas DataFrame.
        target_column (str): Name of the target variable to predict.
        missing_strategy (str): Strategy to handle missing values ('drop' or 'fill').
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Processed features (X) and target (y) ready for modeling.
    """
    # 1. Handle missing values
    df_clean = handle_missing_values(df, strategy=missing_strategy)
    
    # 2. Extract and encode features
    df_encoded = encode_categorical_features(df_clean)
    
    # 3. Separate features and target
    X, y = separate_features_target(df_encoded, target_column=target_column)
    
    return X, y

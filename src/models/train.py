import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate the model using RMSE and R2 Score.
    
    Args:
        y_true (pd.Series): Actual target values.
        y_pred (np.ndarray): Predicted target values.
        
    Returns:
        Dict[str, float]: Dictionary containing RMSE and R2 scores.
    """
    # Calculate Mean Squared Error then take root for RMSE
    # Using np.sqrt(MSE) ensures compatibility across all scikit-learn versions
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        "RMSE": rmse,
        "R2_Score": r2
    }

def train_and_compare_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Train Linear Regression and Decision Tree models and compare their metrics.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing targets.
        
    Returns:
        Dict[str, Any]: Dictionary mapping model names to their trained instances and metrics.
    """
    # 1. Initialize models
    models = {
        "Linear_Regression": LinearRegression(),
        "Decision_Tree": DecisionTreeRegressor(random_state=42)
    }
    
    results = {}
    
    # 2. Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Evaluate predictions
        metrics = evaluate_model(y_test, y_pred)
        
        # Store results
        results[model_name] = {
            "model_instance": model,
            "metrics": metrics
        }
        
    return results

def save_models(results: Dict[str, Any], save_dir: str = "models/") -> None:
    """
    Save the trained models to disk using joblib.
    
    Args:
        results (Dict[str, Any]): Dictionary containing the trained models and metrics.
        save_dir (str): Directory where models will be saved.
    """
    # Ensure directory exists before saving
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, data in results.items():
        model = data["model_instance"]
        file_path = os.path.join(save_dir, f"{model_name}.joblib")
        
        # Serialize model
        joblib.dump(model, file_path)
        print(f"[{model_name}] Saved successfully to: {file_path}")

def display_results(results: Dict[str, Any]) -> None:
    """
    Print the evaluation metrics cleanly for visual comparison.
    """
    print("\n" + "="*45)
    print("           MODEL COMPARISON RESULTS")
    print("="*45)
    
    for model_name, data in results.items():
        metrics = data["metrics"]
        print(f"\nModel: {model_name.replace('_', ' ')}")
        print(f"  - RMSE:     {metrics['RMSE']:.4f}")
        print(f"  - R2 Score: {metrics['R2_Score']:.4f}")
    
    print("\n" + "="*45 + "\n")

def run_training_pipeline(X: pd.DataFrame, y: pd.Series, save_dir: str = "models/", test_size: float = 0.2) -> None:
    """
    Main orchestrator to split data, train models, compare metrics, and save artifacts.
    
    Args:
        X (pd.DataFrame): Processed feature set.
        y (pd.Series): Target variable (e.g. math score).
        save_dir (str): Destination directory for saved `.joblib` model files.
        test_size (float): Proportion of dataset to reserve for testing. Default 0.2.
    """
    print(f"Splitting dataset into train and test sets ({int((1-test_size)*100)}/{int(test_size*100)})...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print("\nStarting Model Training Phase...")
    results = train_and_compare_models(X_train, y_train, X_test, y_test)
    
    display_results(results)
    
    print("Saving Models to disk...")
    save_models(results, save_dir=save_dir)

if __name__ == "__main__":
    # Example execution scaffolding describing how to connect preprocess.py with train.py
    print("This module provides training functions.")
    print("\nTo connect it to your dataset, you can run something like this:")
    print('''
    from src.data_processing.preprocess import run_preprocessing_pipeline
    import pandas as pd
    
    # 1. Load Data
    df = pd.read_csv("data/raw/student_data.csv")
    
    # 2. Preprocess Data 
    # (assuming we want to predict 'math score')
    X, y = run_preprocessing_pipeline(df, target_column='math score', missing_strategy='drop')
    
    # 3. Train and Evaluate
    from src.models.train import run_training_pipeline
    run_training_pipeline(X, y)
    ''')

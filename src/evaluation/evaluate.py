import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from sklearn.metrics import mean_squared_error, r2_score

def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics: RMSE and R2 score.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {"RMSE": rmse, "R2_Score": r2}

def evaluate_models(y_true: pd.Series, predictions_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Evaluate multiple model predictions.
    
    Args:
        y_true (pd.Series): True target values.
        predictions_dict (Dict[str, np.ndarray]): Dictionary mapping model names to their predicted arrays.
        
    Returns:
        pd.DataFrame: DataFrame containing the performance metrics for each model.
    """
    results = []
    
    for model_name, y_pred in predictions_dict.items():
        metrics = calculate_metrics(y_true, y_pred)
        results.append({
            "Model": model_name,
            "RMSE": metrics["RMSE"],
            "R2_Score": metrics["R2_Score"]
        })
        
    metrics_df = pd.DataFrame(results)
    return metrics_df

def plot_model_comparison(metrics_df: pd.DataFrame, save_path: str = "reports/figures/model_comparison.png") -> None:
    """
    Generate and save a bar chart comparing model performance using matplotlib only.
    
    Args:
        metrics_df (pd.DataFrame): Dataframe of calculated model performance metrics.
        save_path (str): Output destination for the figure on disk.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # We have two metrics: RMSE and R2_Score. 
    # Because scales drastically differ, we plot them on side-by-side subplots.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = metrics_df["Model"].tolist()
    rmse_values = metrics_df["RMSE"].tolist()
    r2_values = metrics_df["R2_Score"].tolist()
    
    x = np.arange(len(models))
    width = 0.5
    
    # ------------------
    # 1. RMSE Bar Plot (Lower is better)
    # ------------------
    axes[0].bar(x, rmse_values, width, color='salmon', edgecolor='black', alpha=0.8)
    axes[0].set_ylabel('Root Mean Squared Error')
    axes[0].set_title('RMSE Comparison (Lower is Better)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add numerical value labels atop bars
    for i, v in enumerate(rmse_values):
        axes[0].text(i, v + 0.05 * max(rmse_values), f"{v:.2f}", ha='center', fontweight='bold', color='black')
        
    # ------------------
    # 2. R² Score Plot (Higher is better)
    # ------------------
    axes[1].bar(x, r2_values, width, color='mediumseagreen', edgecolor='black', alpha=0.8)
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('R² Score Comparison (Higher is Better)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha="right")
    axes[1].set_ylim(0, max(1.0, max(r2_values) + 0.1)) # Cap at logical domains
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add numerical value labels atop bars
    for i, v in enumerate(r2_values):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold', color='black')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # Close figure to free memory
    
    print(f"Comparison chart successfully saved to: '{save_path}'")

def plot_feature_importance(model: Any, feature_names: List[str], top_n: int = 10, save_path: str = "reports/figures/feature_importance.png") -> None:
    """
    Extract and plot top feature importances from a tree-based model using matplotlib.
    """
    # Check if model supports feature importances
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {type(model).__name__} does not support feature importances. Skipping.")
        return
        
    importances = model.feature_importances_
    
    # Create DataFrame for easier sorting
    feat_impl_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort and take top N
    feat_impl_df = feat_impl_df.sort_values(by='Importance', ascending=True).tail(top_n)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(feat_impl_df['Feature'], feat_impl_df['Importance'], color='dodgerblue', edgecolor='black', alpha=0.8)
    plt.xlabel('Relative Importance')
    plt.title(f'Top {top_n} Feature Importances ({type(model).__name__})')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add numerical labels
    for index, value in enumerate(feat_impl_df['Importance']):
        plt.text(value + 0.005, index, f"{value:.3f}", va='center', fontweight='bold', color='black')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # Close figure to free memory
    
    print(f"Feature importance chart successfully saved to: '{save_path}'")

def run_evaluation_pipeline(
    y_true: pd.Series, 
    predictions_dict: Dict[str, np.ndarray],
    tree_model: Any = None,
    feature_names: List[str] = None
) -> pd.DataFrame:
    """
    Main controller validating metrics and triggering visualization renders.
    
    Args:
        y_true (pd.Series): Exact true target values from the test set.
        predictions_dict (Dict[str, np.ndarray]): Dictionary of raw predicted variables.
        tree_model (Any, optional): Fitted tree-based model for feature importance.
        feature_names (List[str], optional): Names of features exactly matching training data.
        
    Returns:
        pd.DataFrame: Computed evaluation scores matrix.
    """
    print("Evaluating model predictions...")
    
    # 1. Extract Metrics
    df_metrics = evaluate_models(y_true, predictions_dict)
    
    # 2. Print Summary To Terminal
    print("\n--- Evaluation Results ---")
    print(df_metrics.to_string(index=False))
    print("-" * 26)
    
    # 3. Export Matplotlib Visuals
    print("\nGenerating Matplotlib evaluation comparisons...")
    plot_model_comparison(df_metrics)
    
    # 4. Plot Feature Importance if model provided
    if tree_model is not None and feature_names is not None:
        print("\nExtracting Decision Tree feature importances...")
        plot_feature_importance(tree_model, feature_names)
        
    return df_metrics

if __name__ == "__main__":
    print("Evaluation module initialized.")
    # Example execution testing structure
    # predictions_dict = {
    #     "Linear Regression": lr_model.predict(X_test),
    #     "Decision Tree": dt_model.predict(X_test), 
    #     "Random Forest": rf_model.predict(X_test)
    # }
    # run_evaluation_pipeline(y_test, predictions_dict)

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from sklearn.metrics import mean_squared_error, r2_score


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "R2_Score": r2}


def evaluate_models(y_true: pd.Series, predictions_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Compare predictions against true values for multiple models."""
    results = []
    
    for model_name, y_pred in predictions_dict.items():
        metrics = calculate_metrics(y_true, y_pred)
        results.append({
            "Model": model_name,
            "RMSE": metrics["RMSE"],
            "R2_Score": metrics["R2_Score"]
        })
        
    return pd.DataFrame(results)


def plot_model_comparison(metrics_df: pd.DataFrame, save_path: str = "reports/figures/model_comparison.png") -> None:
    """Generate a side-by-side bar chart for RMSE and R2 scores."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Plot RMSE and R2 on independent subplots due to differing scales
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = metrics_df["Model"].tolist()
    rmse_values = metrics_df["RMSE"].tolist()
    r2_values = metrics_df["R2_Score"].tolist()
    
    x = np.arange(len(models))
    width = 0.5
    
    # RMSE plot
    axes[0].bar(x, rmse_values, width, color='salmon', edgecolor='black', alpha=0.8)
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(rmse_values):
        axes[0].text(i, v + 0.05 * max(rmse_values), f"{v:.2f}", ha='center', fontweight='bold')
        
    # R2 Score plot
    axes[1].bar(x, r2_values, width, color='mediumseagreen', edgecolor='black', alpha=0.8)
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('R² Score Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha="right")
    axes[1].set_ylim(0, max(1.0, max(r2_values) + 0.1)) 
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(r2_values):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved to {save_path}")


def plot_feature_importance(model: Any, feature_names: List[str], top_n: int = 10, save_path: str = "reports/figures/feature_importance.png") -> None:
    """Plot the top N feature importances for a tree-based model."""
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {type(model).__name__} lacks feature importances. Skipping.")
        return
        
    feat_impl_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True).tail(top_n)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feat_impl_df['Feature'], feat_impl_df['Importance'], color='dodgerblue', edgecolor='black', alpha=0.8)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Features ({type(model).__name__})')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    for index, value in enumerate(feat_impl_df['Importance']):
        plt.text(value + 0.005, index, f"{value:.3f}", va='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance saved to {save_path}")


def run_evaluation_pipeline(
    y_true: pd.Series, 
    predictions_dict: Dict[str, np.ndarray],
    tree_model: Any = None,
    feature_names: List[str] = None
) -> pd.DataFrame:
    """Runs the complete evaluation and plotting workflow."""
    print("Evaluating models...")
    df_metrics = evaluate_models(y_true, predictions_dict)
    
    print("\n--- Evaluation Results ---")
    print(df_metrics.to_string(index=False))
    
    plot_model_comparison(df_metrics)
    
    if tree_model is not None and feature_names is not None:
        plot_feature_importance(tree_model, feature_names)
        
    return df_metrics


if __name__ == "__main__":
    pass

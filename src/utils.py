"""
Utility Functions
Exercise 2 - VU Machine Learning WS 2025

This module contains utility functions for data loading, preprocessing, and evaluation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset from a file.
    
    Args:
        filepath: Path to the dataset file
    
    Returns:
        tuple: (X, y) feature matrix and target vector
    """
    # TODO: Implement dataset loading
    # Support different file formats: .csv, .xlsx, .txt
    
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    # TODO: Separate features and target
    # Assuming last column is the target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    return X, y


def preprocess_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the data (handle missing values, normalize, etc.).
    
    Args:
        X: Feature matrix
        y: Target vector
    
    Returns:
        tuple: (X_processed, y_processed)
    """
    # TODO: Implement preprocessing steps
    # - Handle missing values
    # - Normalize/standardize features
    # - Encode categorical variables
    
    # Remove rows with missing values (simple approach)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[mask]
    y_clean = y[mask]
    
    return X_clean, y_clean


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
    
    Returns:
        dict: Dictionary of metric names and values
    """
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    return metrics


def cross_validate(model: Any, X: np.ndarray, y: np.ndarray, 
                   n_splits: int = 5) -> Dict[str, np.ndarray]:
    """
    Perform k-fold cross-validation.
    
    Args:
        model: Model instance with fit() and predict() methods
        X: Feature matrix
        y: Target vector
        n_splits: Number of folds
    
    Returns:
        dict: Dictionary with metric arrays for each fold
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        mse_scores.append(metrics['MSE'])
        rmse_scores.append(metrics['RMSE'])
        mae_scores.append(metrics['MAE'])
        r2_scores.append(metrics['R2'])
        
        print(f"Fold {fold}: MSE={metrics['MSE']:.4f}, R2={metrics['R2']:.4f}")
    
    return {
        'MSE': np.array(mse_scores),
        'RMSE': np.array(rmse_scores),
        'MAE': np.array(mae_scores),
        'R2': np.array(r2_scores)
    }


def print_cv_results(cv_results: Dict[str, np.ndarray]) -> None:
    """
    Print cross-validation results in a formatted way.
    
    Args:
        cv_results: Dictionary of cross-validation results
    """
    print("\n" + "="*50)
    print("Cross-Validation Results")
    print("="*50)
    
    for metric_name, scores in cv_results.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{metric_name}: {mean_score:.4f} (+/- {std_score:.4f})")
    
    print("="*50 + "\n")


def compare_models(results: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    """
    Compare multiple models based on cross-validation results.
    
    Args:
        results: Dictionary mapping model names to their CV results
    
    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = []
    
    for model_name, cv_results in results.items():
        row = {'Model': model_name}
        for metric_name, scores in cv_results.items():
            row[f'{metric_name}_mean'] = np.mean(scores)
            row[f'{metric_name}_std'] = np.std(scores)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save experimental results to a file.
    
    Args:
        results: Dictionary of results
        filepath: Output file path
    """
    # TODO: Implement result saving
    # Can save as JSON, CSV, or pickle
    import json
    
    with open(filepath, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v 
            for k, v in results.items()
        }
        json.dump(results_serializable, f, indent=2)


def main():
    """Test utility functions."""
    # Generate sample data
    X = np.random.randn(100, 5)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
    
    # Test metric calculation
    y_pred = y + np.random.randn(100) * 0.2
    metrics = calculate_metrics(y, y_pred)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()

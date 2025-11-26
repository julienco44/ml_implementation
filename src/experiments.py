"""
Experiments Script
Exercise 2 - VU Machine Learning WS 2025

This script runs the experiments comparing different regression algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import time

from regression_tree import RegressionTree
from random_forest import RandomForest
from utils import (
    load_dataset, preprocess_data, cross_validate, 
    print_cv_results, compare_models, save_results
)


def experiment_single_dataset(X: np.ndarray, y: np.ndarray, 
                              dataset_name: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run experiments on a single dataset.
    
    Args:
        X: Feature matrix
        y: Target vector
        dataset_name: Name of the dataset
    
    Returns:
        dict: Results for all models
    """
    print(f"\n{'='*70}")
    print(f"Running experiments on: {dataset_name}")
    print(f"Dataset shape: {X.shape}")
    print(f"{'='*70}\n")
    
    results = {}
    
    # ==========================================
    # 1. Our Regression Tree
    # ==========================================
    print("1. Testing Our Regression Tree...")
    our_tree = RegressionTree(max_depth=10, min_samples_split=5)
    
    start_time = time.time()
    cv_results_our_tree = cross_validate(our_tree, X, y, n_splits=5)
    train_time_our_tree = time.time() - start_time
    
    results['Our Regression Tree'] = cv_results_our_tree
    print_cv_results(cv_results_our_tree)
    print(f"Training time: {train_time_our_tree:.2f}s\n")
    
    # ==========================================
    # 2. Our Random Forest (different configs)
    # ==========================================
    rf_configs = [
        {'n_trees': 10, 'max_depth': 10, 'name': 'Our RF (10 trees)'},
        {'n_trees': 50, 'max_depth': 10, 'name': 'Our RF (50 trees)'},
        {'n_trees': 100, 'max_depth': 10, 'name': 'Our RF (100 trees)'},
    ]
    
    for config in rf_configs:
        print(f"2. Testing {config['name']}...")
        our_rf = RandomForest(
            n_trees=config['n_trees'],
            max_depth=config['max_depth'],
            random_state=42
        )
        
        start_time = time.time()
        cv_results_our_rf = cross_validate(our_rf, X, y, n_splits=5)
        train_time_our_rf = time.time() - start_time
        
        results[config['name']] = cv_results_our_rf
        print_cv_results(cv_results_our_rf)
        print(f"Training time: {train_time_our_rf:.2f}s\n")
    
    # ==========================================
    # 3. Sklearn Decision Tree
    # ==========================================
    print("3. Testing Sklearn Decision Tree...")
    sklearn_tree = DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42)
    
    start_time = time.time()
    cv_results_sklearn_tree = cross_validate(sklearn_tree, X, y, n_splits=5)
    train_time_sklearn_tree = time.time() - start_time
    
    results['Sklearn Decision Tree'] = cv_results_sklearn_tree
    print_cv_results(cv_results_sklearn_tree)
    print(f"Training time: {train_time_sklearn_tree:.2f}s\n")
    
    # ==========================================
    # 4. Sklearn Random Forest
    # ==========================================
    print("4. Testing Sklearn Random Forest...")
    sklearn_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    start_time = time.time()
    cv_results_sklearn_rf = cross_validate(sklearn_rf, X, y, n_splits=5)
    train_time_sklearn_rf = time.time() - start_time
    
    results['Sklearn Random Forest'] = cv_results_sklearn_rf
    print_cv_results(cv_results_sklearn_rf)
    print(f"Training time: {train_time_sklearn_rf:.2f}s\n")
    
    # ==========================================
    # 5. Linear Regression (baseline)
    # ==========================================
    print("5. Testing Linear Regression (baseline)...")
    linear_model = LinearRegression()
    
    start_time = time.time()
    cv_results_linear = cross_validate(linear_model, X, y, n_splits=5)
    train_time_linear = time.time() - start_time
    
    results['Linear Regression'] = cv_results_linear
    print_cv_results(cv_results_linear)
    print(f"Training time: {train_time_linear:.2f}s\n")
    
    # ==========================================
    # Compare all models
    # ==========================================
    print("\nComparison Table:")
    comparison_df = compare_models(results)
    print(comparison_df.to_string(index=False))
    
    return results


def run_all_experiments():
    """
    Run experiments on all datasets.
    """
    # TODO: Update these paths with your actual dataset paths
    datasets = [
        {'name': 'Dataset 1', 'path': 'data/dataset1.csv'},
        {'name': 'Dataset 2', 'path': 'data/dataset2.csv'},
        {'name': 'Dataset 3', 'path': 'data/dataset3.csv'},
    ]
    
    all_results = {}
    
    for dataset in datasets:
        try:
            # Load and preprocess data
            print(f"\nLoading {dataset['name']}...")
            X, y = load_dataset(dataset['path'])
            X, y = preprocess_data(X, y)
            
            # Run experiments
            results = experiment_single_dataset(X, y, dataset['name'])
            all_results[dataset['name']] = results
            
            # Save results
            save_results(results, f"results/metrics/{dataset['name']}_results.json")
            
        except FileNotFoundError:
            print(f"Warning: Could not find {dataset['path']}")
            print("Please download the datasets first and update the paths.")
        except Exception as e:
            print(f"Error processing {dataset['name']}: {str(e)}")
    
    return all_results


def main():
    """
    Main function to run all experiments.
    """
    print("="*70)
    print("Machine Learning Exercise 2 - Experiments")
    print("Regression Trees and Random Forests")
    print("="*70)
    
    # Run all experiments
    all_results = run_all_experiments()
    
    print("\n" + "="*70)
    print("All experiments completed!")
    print("Results saved in: results/metrics/")
    print("="*70)


if __name__ == "__main__":
    main()

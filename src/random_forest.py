"""
Random Forest Implementation
Exercise 2 - VU Machine Learning WS 2025

This module implements a random forest algorithm based on the regression tree implementation.
"""

import numpy as np
from typing import Optional
from regression_tree import RegressionTree


class RandomForest:
    """
    Random Forest Regressor implementation from scratch.
    
    Parameters:
        n_trees: Number of trees in the forest
        max_depth: Maximum depth of each tree
        min_samples_split: Minimum number of samples required to split a node
        min_samples_leaf: Minimum number of samples required in a leaf node
        max_features: Number of features to consider for each tree
        bootstrap: Whether to use bootstrap sampling
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, n_trees: int = 100,
                 max_depth: int = 10,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[int] = None,
                 bootstrap: bool = True,
                 random_state: Optional[int] = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForest':
        """
        Build a forest of regression trees from training data.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        
        Returns:
            self: Fitted regressor
        """
        np.random.seed(self.random_state)
        self.trees = []
        
        # TODO: Set max_features if not specified
        # Common choice: max_features = sqrt(n_features) or n_features/3
        
        n_samples, n_features = X.shape
        
        # Build each tree in the forest
        for i in range(self.n_trees):
            # TODO: Create bootstrap sample if bootstrap=True
            # Otherwise use the full dataset
            
            if self.bootstrap:
                # Bootstrap sampling: sample with replacement
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample = X
                y_sample = y
            
            # TODO: Create and train a regression tree
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples in X.
        
        Args:
            X: Samples of shape (n_samples, n_features)
        
        Returns:
            np.ndarray: Predicted values of shape (n_samples,)
        """
        # TODO: Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # TODO: Average predictions across all trees
        predictions = np.mean(tree_predictions, axis=0)
        
        return predictions
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Calculate feature importance based on how often features are used.
        
        Returns:
            np.ndarray: Feature importance scores
        """
        # TODO: Implement feature importance calculation
        # Can be based on:
        # - Frequency of feature usage
        # - Total variance reduction
        # - Other metrics
        pass


def main():
    """Example usage of RandomForest."""
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
    
    # Train random forest with different configurations
    configs = [
        {"n_trees": 10, "max_depth": 5},
        {"n_trees": 50, "max_depth": 5},
        {"n_trees": 100, "max_depth": 5},
    ]
    
    for config in configs:
        print(f"\nConfiguration: {config}")
        rf = RandomForest(**config, random_state=42)
        rf.fit(X, y)
        
        # Make predictions
        predictions = rf.predict(X)
        
        # Calculate MSE
        mse = np.mean((y - predictions) ** 2)
        print(f"Training MSE: {mse:.4f}")


if __name__ == "__main__":
    main()

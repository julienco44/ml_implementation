"""
Regression Tree Implementation
Exercise 2 - VU Machine Learning WS 2025

This module implements a regression tree algorithm from scratch.
"""

import numpy as np
from typing import Optional, Union


class Node:
    """
    A node in the regression tree.
    
    Attributes:
        feature_idx: Index of the feature to split on (None for leaf nodes)
        threshold: Threshold value for splitting (None for leaf nodes)
        left: Left child node
        right: Right child node
        value: Predicted value for leaf nodes (None for internal nodes)
    """
    
    def __init__(self, feature_idx: Optional[int] = None, 
                 threshold: Optional[float] = None,
                 left: Optional['Node'] = None,
                 right: Optional['Node'] = None,
                 value: Optional[float] = None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf node."""
        return self.value is not None


class RegressionTree:
    """
    Regression Tree implementation from scratch.
    
    Parameters:
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum number of samples required to split a node
        min_samples_leaf: Minimum number of samples required in a leaf node
        max_features: Number of features to consider when looking for the best split
    """
    
    def __init__(self, max_depth: int = 10, 
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RegressionTree':
        """
        Build the regression tree from training data.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        
        Returns:
            self: Fitted regressor
        """
        self.n_features_ = X.shape[1]
        if self.max_features is None:
            self.max_features = self.n_features_
        
        # Build the tree recursively
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Recursively build the regression tree.
        
        Args:
            X: Feature matrix
            y: Target values
            depth: Current depth in the tree
        
        Returns:
            Node: Root node of the (sub)tree
        """
        n_samples, n_features = X.shape
        
        # TODO: Implement stopping criteria
        # - Check max_depth
        # - Check min_samples_split
        # - Check if all y values are the same
        
        # TODO: If stopping criteria met, create leaf node
        # leaf_value = self._calculate_leaf_value(y)
        # return Node(value=leaf_value)
        
        # TODO: Find the best split
        # best_feature, best_threshold = self._find_best_split(X, y)
        
        # TODO: If no valid split found, create leaf node
        
        # TODO: Split the data
        # left_indices = X[:, best_feature] <= best_threshold
        # right_indices = ~left_indices
        
        # TODO: Recursively build left and right subtrees
        # left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        # right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        # TODO: Return internal node
        # return Node(feature_idx=best_feature, threshold=best_threshold,
        #             left=left_child, right=right_child)
        
        # Placeholder
        return Node(value=np.mean(y))
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Find the best feature and threshold to split on.
        
        Args:
            X: Feature matrix
            y: Target values
        
        Returns:
            tuple: (best_feature_idx, best_threshold)
        """
        # TODO: Implement split finding logic
        # Consider using variance reduction or MSE reduction
        # Try different splitting criteria
        
        best_feature = None
        best_threshold = None
        best_score = float('inf')
        
        # TODO: Iterate over features
        # TODO: For each feature, iterate over possible thresholds
        # TODO: Calculate the quality of split (e.g., MSE reduction)
        # TODO: Keep track of the best split
        
        return best_feature, best_threshold
    
    def _calculate_split_score(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Calculate the score of a split (lower is better).
        
        Args:
            y_left: Target values in left split
            y_right: Target values in right split
        
        Returns:
            float: Split score (e.g., weighted MSE)
        """
        # TODO: Implement split scoring
        # Common approach: weighted average of variance or MSE
        pass
    
    def _calculate_leaf_value(self, y: np.ndarray) -> float:
        """
        Calculate the prediction value for a leaf node.
        
        Args:
            y: Target values in the leaf
        
        Returns:
            float: Predicted value (typically mean)
        """
        return np.mean(y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples in X.
        
        Args:
            X: Samples of shape (n_samples, n_features)
        
        Returns:
            np.ndarray: Predicted values of shape (n_samples,)
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x: np.ndarray, node: Node) -> float:
        """
        Traverse the tree to make a prediction for a single sample.
        
        Args:
            x: Single sample
            node: Current node
        
        Returns:
            float: Predicted value
        """
        if node.is_leaf():
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


def main():
    """Example usage of RegressionTree."""
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
    
    # Train regression tree
    tree = RegressionTree(max_depth=5, min_samples_split=5)
    tree.fit(X, y)
    
    # Make predictions
    predictions = tree.predict(X)
    
    # Calculate MSE
    mse = np.mean((y - predictions) ** 2)
    print(f"Training MSE: {mse:.4f}")


if __name__ == "__main__":
    main()


from typing import Optional, Tuple
import numpy as np


class Node:
    """
    A node in the regression tree.
    
    Attributes:
        feature_idx: Index of the feature to split on (None for leaf nodes)
        threshold: Threshold value for splitting (None for leaf nodes)
        left: Left child node (samples with feature <= threshold)
        right: Right child node (samples with feature > threshold)
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
    
    Uses CART algorithm with variance reduction (weighted MSE) as splitting criterion.
    Compatible with sklearn's API (fit/predict interface).
    """
    
    def __init__(self, max_depth: int = 10, 
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root: Optional[Node] = None
        self.n_features_: Optional[int] = None
        self.max_features_: Optional[int] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RegressionTree':
        """Build the regression tree from training data."""
        self.n_features_ = X.shape[1]
        if self.max_features is None:
            self.max_features_ = self.n_features_
        else:
            self.max_features_ = min(self.max_features, self.n_features_)
        
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Recursively build the regression tree."""
        n_samples, n_features = X.shape
        
        # ----- STOPPING CONDITIONS -----
        if depth >= self.max_depth:
            return Node(value=self._calculate_leaf_value(y))
        
        if n_samples < self.min_samples_split:
            return Node(value=self._calculate_leaf_value(y))
        
        if np.std(y) < 1e-10:
            return Node(value=self._calculate_leaf_value(y))
        
        # ----- FIND BEST SPLIT -----
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            return Node(value=self._calculate_leaf_value(y))
        
        # ----- SPLIT DATA -----
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        if (np.sum(left_indices) < self.min_samples_leaf or 
            np.sum(right_indices) < self.min_samples_leaf):
            return Node(value=self._calculate_leaf_value(y))
        
        # ----- CREATE CHILDREN -----
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature_idx=best_feature, threshold=best_threshold,
                    left=left_child, right=right_child)
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """Find the best feature and threshold to split on."""
        best_feature = None
        best_threshold = None
        best_score = float('inf')
        
        n_samples, n_features = X.shape
        
        # random subset of features if max_features_ < n_features
        if self.max_features_ < n_features:
            feature_indices = np.random.choice(n_features, self.max_features_, replace=False)
        else:
            feature_indices = np.arange(n_features)
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            if len(unique_values) < 2:
                continue
            
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if (np.sum(left_mask) < self.min_samples_leaf or 
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue
                
                score = self._calculate_split_score(y[left_mask], y[right_mask])
                
                if score < best_score:
                    best_score = score
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _calculate_split_score(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Weighted variance (MSE) of a split (lower = better)."""
        n_left, n_right = len(y_left), len(y_right)
        n_total = n_left + n_right
        
        mse_left = np.var(y_left) if n_left > 0 else 0.0
        mse_right = np.var(y_right) if n_right > 0 else 0.0
        
        return (n_left / n_total) * mse_left + (n_right / n_total) * mse_right
    
    def _calculate_leaf_value(self, y: np.ndarray) -> float:
        """Prediction value for a leaf node: mean of target values."""
        return float(np.mean(y))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for samples in X."""
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x: np.ndarray, node: Node) -> float:
        """Traverse the tree to make a prediction for a single sample."""
        if node.is_leaf():
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


print("✓ RegressionTree class defined")

"""
Regression Tree Implementation with Cost-Complexity Pruning (CCP)
Exercise 2 - VU Machine Learning WS 2025

This module implements a regression tree algorithm from scratch with:
- CART algorithm (variance reduction criterion)
- Cost-Complexity Pruning (post-pruning)
- Full sklearn-compatible API
"""

import numpy as np
from typing import Optional, Tuple, List
import copy


class Node:
    """
    A node in the regression tree.
    
    Attributes:
        feature_idx: Index of the feature to split on (None for leaf nodes)
        threshold: Threshold value for splitting (None for leaf nodes)
        left: Left child node (samples with feature <= threshold)
        right: Right child node (samples with feature > threshold)
        value: Predicted value for leaf nodes (None for internal nodes)
        n_samples: Number of samples at this node (for pruning)
        impurity: Impurity (MSE) at this node (for pruning)
        
    Cached values for fast CCP:
        _n_leaves: Cached leaf count
        _subtree_impurity: Cached subtree impurity
        _effective_alpha: Cached effective alpha
    """
    
    def __init__(self, feature_idx: Optional[int] = None, 
                 threshold: Optional[float] = None,
                 left: Optional['Node'] = None,
                 right: Optional['Node'] = None,
                 value: Optional[float] = None,
                 n_samples: int = 0,
                 impurity: float = 0.0):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.n_samples = n_samples
        self.impurity = impurity
        # Cached values for fast CCP
        self._n_leaves = None
        self._subtree_impurity = None
        self._effective_alpha = None
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf node."""
        return self.value is not None


class RegressionTree:
    """
    Regression Tree implementation from scratch with Cost-Complexity Pruning.
    
    Uses CART algorithm with variance reduction (weighted MSE) as splitting criterion.
    Compatible with sklearn's API (fit/predict interface).
    
    Parameters:
        max_depth: Maximum depth of the tree (default=None, no limit)
        min_samples_split: Minimum samples required to split a node (default=2)
        min_samples_leaf: Minimum samples required in a leaf node (default=1)
        max_features: Number of features to consider for best split (default=None, uses all)
        ccp_alpha: Complexity parameter for Cost-Complexity Pruning (default=0.0, no pruning)
    
    Attributes:
        root: Root node of the fitted tree
        n_features_: Number of features in training data
        ccp_alphas_: List of effective alphas for pruning path (after fit)
        impurities_: List of total impurities for pruning path (after fit)
    """
    
    def __init__(self, max_depth: Optional[int] = None, 
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[int] = None,
                 ccp_alpha: float = 0.0):
        self.max_depth = max_depth if max_depth is not None else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.ccp_alpha = ccp_alpha
        self.root = None
        self.n_features_ = None
        self.ccp_alphas_ = []
        self.impurities_ = []
    
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
            self.max_features_ = self.n_features_
        else:
            self.max_features_ = min(self.max_features, self.n_features_)
        
        # Build the full tree
        self.root = self._build_tree(X, y, depth=0)
        
        # Apply Cost-Complexity Pruning if alpha > 0
        if self.ccp_alpha > 0:
            self._prune_tree(self.ccp_alpha)
        
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
        # Weighted impurity: n * var(y) = sum of squared errors
        # This ensures subtree impurities sum correctly for CCP
        impurity = n_samples * np.var(y)
        
        # ========== STOPPING CONDITIONS ==========
        
        # 1. Max depth reached
        if depth >= self.max_depth:
            return Node(value=self._calculate_leaf_value(y), 
                       n_samples=n_samples, impurity=impurity)
        
        # 2. Not enough samples to split
        if n_samples < self.min_samples_split:
            return Node(value=self._calculate_leaf_value(y),
                       n_samples=n_samples, impurity=impurity)
        
        # 3. Pure node (all target values nearly identical)
        if np.std(y) < 1e-10:
            return Node(value=self._calculate_leaf_value(y),
                       n_samples=n_samples, impurity=impurity)
        
        # ========== FIND BEST SPLIT ==========
        best_feature, best_threshold = self._find_best_split(X, y)
        
        # No valid split found
        if best_feature is None:
            return Node(value=self._calculate_leaf_value(y),
                       n_samples=n_samples, impurity=impurity)
        
        # ========== SPLIT DATA ==========
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        # Check min_samples_leaf constraint
        if (np.sum(left_indices) < self.min_samples_leaf or 
            np.sum(right_indices) < self.min_samples_leaf):
            return Node(value=self._calculate_leaf_value(y),
                       n_samples=n_samples, impurity=impurity)
        
        # ========== CREATE CHILDREN ==========
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature_idx=best_feature, threshold=best_threshold,
                    left=left_child, right=right_child,
                    n_samples=n_samples, impurity=impurity)
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """
        Find the best feature and threshold to split on.
        Uses variance reduction (minimizing weighted MSE).
        
        Args:
            X: Feature matrix
            y: Target values
        
        Returns:
            tuple: (best_feature_idx, best_threshold)
        """
        best_feature = None
        best_threshold = None
        best_score = float('inf')
        
        n_samples, n_features = X.shape
        
        # Select features to consider (random subset for Random Forest)
        if self.max_features_ < n_features:
            feature_indices = np.random.choice(n_features, self.max_features_, replace=False)
        else:
            feature_indices = np.arange(n_features)
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # Skip if only one unique value
            if len(unique_values) < 2:
                continue
            
            # Midpoints between unique values as thresholds
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # Skip invalid splits
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
        """
        Calculate weighted MSE of a split (lower = better).
        
        This is the variance reduction criterion for regression trees.
        
        Args:
            y_left: Target values in left split
            y_right: Target values in right split
        
        Returns:
            float: Weighted variance (split score)
        """
        n_left, n_right = len(y_left), len(y_right)
        n_total = n_left + n_right
        
        mse_left = np.var(y_left) if n_left > 0 else 0
        mse_right = np.var(y_right) if n_right > 0 else 0
        
        return (n_left / n_total) * mse_left + (n_right / n_total) * mse_right
    
    def _calculate_leaf_value(self, y: np.ndarray) -> float:
        """
        Calculate the prediction value for a leaf node.
        
        For regression, this is the mean of target values.
        
        Args:
            y: Target values in the leaf
        
        Returns:
            float: Predicted value (mean)
        """
        return np.mean(y)
    
    # ========== COST-COMPLEXITY PRUNING ==========
    
    def _prune_tree(self, alpha: float) -> None:
        """
        Apply Cost-Complexity Pruning to the tree.
        
        Prunes subtrees where the effective alpha is less than the given alpha.
        
        Args:
            alpha: Complexity parameter (higher = more pruning)
        """
        self._prune_subtree(self.root, alpha)
    
    def _prune_subtree(self, node: Node, alpha: float) -> Node:
        """
        Recursively prune a subtree.
        
        Args:
            node: Current node
            alpha: Complexity parameter
        
        Returns:
            Node: Pruned node (may be converted to leaf)
        """
        if node.is_leaf():
            return node
        
        # First, recursively prune children
        node.left = self._prune_subtree(node.left, alpha)
        node.right = self._prune_subtree(node.right, alpha)
        
        # Calculate effective alpha for this subtree
        n_leaves = self._count_leaves(node)
        subtree_impurity = self._get_subtree_impurity(node)
        
        # R(t) - impurity if this node were a leaf
        node_impurity = node.impurity
        
        # Calculate effective alpha: (R(t) - R(Tt)) / (|Tt| - 1)
        if n_leaves > 1:
            effective_alpha = (node_impurity - subtree_impurity) / (n_leaves - 1)
        else:
            effective_alpha = float('inf')
        
        # Prune if effective alpha <= given alpha
        if effective_alpha <= alpha:
            # Convert to leaf
            leaf_value = self._get_subtree_mean(node)
            return Node(value=leaf_value, n_samples=node.n_samples, impurity=node_impurity)
        
        return node
    
    def _count_leaves(self, node: Node) -> int:
        """Count the number of leaves in a subtree (with caching)."""
        if node.is_leaf():
            return 1
        if node._n_leaves is not None:
            return node._n_leaves
        node._n_leaves = self._count_leaves(node.left) + self._count_leaves(node.right)
        return node._n_leaves
    
    def _get_subtree_impurity(self, node: Node) -> float:
        """Get the total impurity of leaves in a subtree (with caching)."""
        if node.is_leaf():
            return node.impurity
        if node._subtree_impurity is not None:
            return node._subtree_impurity
        node._subtree_impurity = self._get_subtree_impurity(node.left) + self._get_subtree_impurity(node.right)
        return node._subtree_impurity
    
    def _get_subtree_mean(self, node: Node) -> float:
        """Get the weighted mean prediction of a subtree."""
        if node.is_leaf():
            return node.value
        
        # Weighted average of children
        left_val = self._get_subtree_mean(node.left)
        right_val = self._get_subtree_mean(node.right)
        
        total_n = node.n_samples
        if total_n == 0:
            return 0
        left_n = node.left.n_samples if node.left else 0
        right_n = node.right.n_samples if node.right else 0
        return (left_val * left_n + right_val * right_n) / total_n
    
    def _cache_tree_stats(self, node: Node) -> None:
        """Pre-compute and cache all tree statistics in ONE pass (bottom-up)."""
        if node is None:
            return
        
        if node.is_leaf():
            node._n_leaves = 1
            node._subtree_impurity = node.impurity
            node._effective_alpha = float('inf')
            return
        
        # Recurse first (bottom-up)
        self._cache_tree_stats(node.left)
        self._cache_tree_stats(node.right)
        
        # Now compute this node's values
        node._n_leaves = node.left._n_leaves + node.right._n_leaves
        node._subtree_impurity = node.left._subtree_impurity + node.right._subtree_impurity
        
        # Effective alpha
        if node._n_leaves > 1:
            node._effective_alpha = (node.impurity - node._subtree_impurity) / (node._n_leaves - 1)
        else:
            node._effective_alpha = float('inf')
    
    def cost_complexity_pruning_path(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Compute the pruning path using Minimal Cost-Complexity Pruning.
        
        Builds ONE tree and iteratively prunes weakest links WITHOUT retraining.
        Records (alpha, impurity) at each pruning step.
        
        Args:
            X: Training features
            y: Target values
        
        Returns:
            dict: {'ccp_alphas': array of alpha values, 
                   'impurities': array of total impurities}
        """
        # Save original settings
        original_max_depth = self.max_depth
        original_ccp_alpha = self.ccp_alpha
        
        # Build ONE full tree
        self.max_depth = 12
        self.ccp_alpha = 0.0
        self.fit(X, y)
        
        # Make a deep copy to prune (keep original intact)
        pruning_tree = copy.deepcopy(self.root)
        
        # Initialize with full tree state
        self._cache_tree_stats(pruning_tree)
        
        alphas = [0.0]
        impurities = [pruning_tree._subtree_impurity]
        
        # Iteratively find and prune weakest links
        while not pruning_tree.is_leaf():
            # Find weakest link (smallest effective alpha)
            weakest_node, weakest_alpha = self._find_weakest_link_node(pruning_tree)
            
            if weakest_node is None or weakest_alpha == float('inf'):
                break
            
            # Prune the weakest node (convert to leaf)
            self._prune_node_inplace(weakest_node)
            
            # Re-cache stats after pruning
            self._cache_tree_stats(pruning_tree)
            
            # Record this state
            alphas.append(weakest_alpha)
            impurities.append(pruning_tree._subtree_impurity if not pruning_tree.is_leaf() 
                            else pruning_tree.impurity)
        
        # Restore original settings
        self.max_depth = original_max_depth
        self.ccp_alpha = original_ccp_alpha
        
        self.ccp_alphas_ = np.array(alphas)
        self.impurities_ = np.array(impurities)
        
        return {'ccp_alphas': self.ccp_alphas_, 'impurities': self.impurities_}
    
    def _find_weakest_link_node(self, node: Node) -> Tuple[Optional[Node], float]:
        """Find the node with smallest effective alpha in the tree."""
        if node is None or node.is_leaf():
            return None, float('inf')
        
        # This node's effective alpha
        current_alpha = node._effective_alpha if node._effective_alpha is not None else float('inf')
        weakest_node = node
        weakest_alpha = current_alpha
        
        # Check left subtree
        left_node, left_alpha = self._find_weakest_link_node(node.left)
        if left_alpha < weakest_alpha:
            weakest_node = left_node
            weakest_alpha = left_alpha
        
        # Check right subtree
        right_node, right_alpha = self._find_weakest_link_node(node.right)
        if right_alpha < weakest_alpha:
            weakest_node = right_node
            weakest_alpha = right_alpha
        
        return weakest_node, weakest_alpha
    
    def _prune_node_inplace(self, node: Node) -> None:
        """Convert an internal node to a leaf (in-place modification)."""
        if node is None or node.is_leaf():
            return
        
        # Calculate leaf value as weighted mean of subtree
        leaf_value = self._get_subtree_mean(node)
        
        # Convert to leaf by setting value and removing children
        node.value = leaf_value
        node.left = None
        node.right = None
        node.feature_idx = None
        node.threshold = None
        # Reset cached values
        node._n_leaves = 1
        node._subtree_impurity = node.impurity
        node._effective_alpha = float('inf')
    
    # ========== PREDICTION ==========
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples in X.

        Optimized: Uses vectorized batch prediction instead of per-sample loops.
        
        Args:
            X: Samples of shape (n_samples, n_features)
        
        Returns:
            np.ndarray: Predicted values of shape (n_samples,)
        """
        return self._predict_batch(X, self.root)

    def _predict_batch(self, X: np.ndarray, node: Node) -> np.ndarray:
        """
        Vectorized batch prediction - processes all samples at once per node.

        ~5-10x faster than per-sample traversal for large datasets.
        """
        n_samples = X.shape[0]

        if node.is_leaf():
            return np.full(n_samples, node.value)

        # Split samples based on this node's condition
        left_mask = X[:, node.feature_idx] <= node.threshold
        right_mask = ~left_mask

        # Allocate result array
        predictions = np.empty(n_samples)

        # Recursively predict for each branch
        if np.any(left_mask):
            predictions[left_mask] = self._predict_batch(X[left_mask], node.left)
        if np.any(right_mask):
            predictions[right_mask] = self._predict_batch(X[right_mask], node.right)

        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return R² score for predictions on X against y.

        sklearn-compatible scoring method.
        
        Args:
            X: Samples of shape (n_samples, n_features)
            y: True target values
        
        Returns:
            float: R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # ========== TREE ANALYSIS ==========
    
    def get_depth(self) -> int:
        """Get the depth of the tree."""
        return self._get_depth(self.root)
    
    def _get_depth(self, node: Node, current_depth: int = 0) -> int:
        if node is None or node.is_leaf():
            return current_depth
        return max(self._get_depth(node.left, current_depth + 1),
                   self._get_depth(node.right, current_depth + 1))
    
    def get_n_leaves(self) -> int:
        """Get the number of leaves in the tree."""
        return self._count_leaves(self.root)
    
    def get_n_nodes(self) -> int:
        """Get the total number of nodes in the tree."""
        return self._count_nodes(self.root)
    
    def _count_nodes(self, node: Node) -> int:
        if node is None:
            return 0
        if node.is_leaf():
            return 1
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)
    
    def get_feature_importances(self) -> np.ndarray:
        """
        Compute feature importances based on impurity decrease.
        
        Returns:
            np.ndarray: Feature importances (normalized to sum to 1)
        """
        importances = np.zeros(self.n_features_)
        self._compute_feature_importances(self.root, importances)
        
        # Normalize
        total = importances.sum()
        if total > 0:
            importances /= total
        
        return importances
    
    def _compute_feature_importances(self, node: Node, importances: np.ndarray) -> None:
        """Recursively compute feature importances."""
        if node is None or node.is_leaf():
            return
        
        # Impurity decrease at this node
        left_impurity = self._get_subtree_impurity(node.left)
        right_impurity = self._get_subtree_impurity(node.right)
        impurity_decrease = node.impurity - left_impurity - right_impurity
        
        importances[node.feature_idx] += impurity_decrease
        
        self._compute_feature_importances(node.left, importances)
        self._compute_feature_importances(node.right, importances)


# ========== OPTIMIZED VERSION WITH PRE-SORTED FEATURES ==========

class OptimizedRegressionTree(RegressionTree):
    """
    Optimized Regression Tree with pre-sorted features.
    
    This version pre-sorts all features ONCE at the beginning,
    making split finding O(n) instead of O(n log n) per feature.
    
    ~3-5x faster than the base RegressionTree for medium/large datasets.
    """
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OptimizedRegressionTree':
        """
        Build the regression tree with pre-sorted features.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        
        Returns:
            self: Fitted regressor
        """
        self.n_features_ = X.shape[1]
        if self.max_features is None:
            self.max_features_ = self.n_features_
        else:
            self.max_features_ = min(self.max_features, self.n_features_)
        
        # ===== PRE-SORT all features ONCE =====
        self.sorted_indices = np.empty((X.shape[0], X.shape[1]), dtype=np.int64)
        
        for feature_idx in range(X.shape[1]):
            self.sorted_indices[:, feature_idx] = np.argsort(X[:, feature_idx])
        
        # Build tree with sorted data
        sample_indices = np.arange(len(y))
        self.root = self._build_tree_sorted(X, y, sample_indices, depth=0)
        
        # Apply Cost-Complexity Pruning if alpha > 0
        if self.ccp_alpha > 0:
            self._prune_tree(self.ccp_alpha)

        return self

    def _build_tree_sorted(self, X: np.ndarray, y: np.ndarray,
                           sample_indices: np.ndarray, depth: int) -> Node:
        """Build tree using pre-sorted features."""
        n_samples = len(sample_indices)
        y_subset = y[sample_indices]
        # Weighted impurity: n * var(y) = sum of squared errors
        # This ensures subtree impurities sum correctly for CCP
        impurity = n_samples * np.var(y_subset)
        
        # ========== STOPPING CONDITIONS ==========
        if depth >= self.max_depth:
            return Node(value=np.mean(y_subset), n_samples=n_samples, impurity=impurity)
        
        if n_samples < self.min_samples_split:
            return Node(value=np.mean(y_subset), n_samples=n_samples, impurity=impurity)
        
        if np.std(y_subset) < 1e-10:
            return Node(value=np.mean(y_subset), n_samples=n_samples, impurity=impurity)
        
        # ========== FIND BEST SPLIT using sorted data ==========
        best_feature, best_threshold, left_indices, right_indices = \
            self._find_best_split_sorted(X, y, sample_indices)
        
        if best_feature is None:
            return Node(value=np.mean(y_subset), n_samples=n_samples, impurity=impurity)
        
        # Check min_samples_leaf
        if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
            return Node(value=np.mean(y_subset), n_samples=n_samples, impurity=impurity)
        
        # Recursively build children
        left_child = self._build_tree_sorted(X, y, left_indices, depth + 1)
        right_child = self._build_tree_sorted(X, y, right_indices, depth + 1)
        
        return Node(feature_idx=best_feature, threshold=best_threshold,
                    left=left_child, right=right_child,
                    n_samples=n_samples, impurity=impurity)
    
    def _find_best_split_sorted(self, X: np.ndarray, y: np.ndarray, 
                                sample_indices: np.ndarray) -> Tuple[Optional[int], Optional[float], 
                                                                      Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find best split using pre-sorted features + INCREMENTAL VARIANCE.
        
        Optimizations:
        1. Pre-sorted features - no sorting per split
        2. Incremental variance - O(1) variance update instead of O(n)
        
        This is ~10x faster than the naive approach.
        """
        best_feature = None
        best_threshold = None
        best_score = float('inf')
        best_left_indices = None
        best_right_indices = None
        
        n_features = X.shape[1]
        
        if self.max_features_ < n_features:
            feature_indices = np.random.choice(n_features, self.max_features_, replace=False)
        else:
            feature_indices = np.arange(n_features)
        
        # Create boolean mask for O(1) lookup (CRITICAL for performance!)
        n_total = len(self.sorted_indices)
        in_sample = np.zeros(n_total, dtype=bool)
        in_sample[sample_indices] = True
        
        for feature_idx in feature_indices:
            # Get sorted indices for this feature
            global_sorted_idx = self.sorted_indices[:, feature_idx]
            
            # Filter using boolean mask - very fast!
            mask = in_sample[global_sorted_idx]
            sorted_positions = global_sorted_idx[mask]
            
            n = len(sorted_positions)
            if n < 2 * self.min_samples_leaf:
                continue
            
            sorted_feature_values = X[sorted_positions, feature_idx]
            sorted_y = y[sorted_positions]
            
            # ===== INCREMENTAL VARIANCE using running sums =====
            # Variance formula: Var = E[X²] - E[X]² = (sum_sq/n) - (sum/n)²
            
            # Initialize: first min_samples_leaf on left, rest on right
            init_left = self.min_samples_leaf
            
            sum_left = np.sum(sorted_y[:init_left])
            sum_sq_left = np.sum(sorted_y[:init_left] ** 2)
            n_left = init_left
            
            sum_right = np.sum(sorted_y[init_left:])
            sum_sq_right = np.sum(sorted_y[init_left:] ** 2)
            n_right = n - init_left
            
            # Iterate through possible split points
            for i in range(init_left, n - self.min_samples_leaf):
                # Only split when value actually changes
                if sorted_feature_values[i] == sorted_feature_values[i-1]:
                    # Still need to update running sums
                    val = sorted_y[i]
                    sum_left += val
                    sum_sq_left += val * val
                    sum_right -= val
                    sum_sq_right -= val * val
                    n_left += 1
                    n_right -= 1
                    continue
                
                # Calculate variances from running sums (O(1)!)
                mean_left = sum_left / n_left
                mean_right = sum_right / n_right
                
                var_left = (sum_sq_left / n_left) - (mean_left * mean_left)
                var_right = (sum_sq_right / n_right) - (mean_right * mean_right)
                
                # Handle numerical issues
                var_left = max(0, var_left)
                var_right = max(0, var_right)
    
                # Weighted variance score
                score = (n_left * var_left + n_right * var_right) / n
                
                if score < best_score:
                    best_score = score
                    best_feature = feature_idx
                    best_threshold = (sorted_feature_values[i] + sorted_feature_values[i-1]) / 2
                    best_split_idx = i
                    best_sorted_positions = sorted_positions
                
                # Update running sums for next iteration
                val = sorted_y[i]
                sum_left += val
                sum_sq_left += val * val
                sum_right -= val
                sum_sq_right -= val * val
                n_left += 1
                n_right -= 1
        
        # Get actual indices for best split
        if best_feature is not None:
            best_left_indices = best_sorted_positions[:best_split_idx]
            best_right_indices = best_sorted_positions[best_split_idx:]
        
        return best_feature, best_threshold, best_left_indices, best_right_indices


# ========== CONVENIENCE FUNCTIONS ==========

def print_tree(node: Node, depth: int = 0, max_depth: int = 3, 
               feature_names: List[str] = None) -> None:
    """
    Print tree structure up to max_depth.
    
    Args:
        node: Root node
        depth: Current depth
        max_depth: Maximum depth to print
        feature_names: List of feature names
    """
    indent = "  " * depth
    
    if node.is_leaf():
        print(f"{indent}→ Leaf: {node.value:.4f} (n={node.n_samples})")
    elif depth >= max_depth:
        print(f"{indent}...")
    else:
        fname = feature_names[node.feature_idx] if feature_names else f"X[{node.feature_idx}]"
        print(f"{indent}[{fname} <= {node.threshold:.4f}] (n={node.n_samples})")
        print(f"{indent}├─ True:")
        print_tree(node.left, depth + 1, max_depth, feature_names)
        print(f"{indent}└─ False:")
        print_tree(node.right, depth + 1, max_depth, feature_names)


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    
    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("=" * 60)
    print("Regression Tree Demo")
    print("=" * 60)
    
    # OptimizedRegressionTree
    tree = OptimizedRegressionTree(max_depth=10)
    tree.fit(X_train, y_train)
    print(f"Depth: {tree.get_depth()}, Leaves: {tree.get_n_leaves()}")
    print(f"Test R²: {r2_score(y_test, tree.predict(X_test)):.4f}")
    
    # With CCP
    print("\nWith Cost-Complexity Pruning (alpha=100):")
    tree_ccp = OptimizedRegressionTree(max_depth=15, ccp_alpha=100)
    tree_ccp.fit(X_train, y_train)
    print(f"Depth: {tree_ccp.get_depth()}, Leaves: {tree_ccp.get_n_leaves()}")
    print(f"Test R²: {r2_score(y_test, tree_ccp.predict(X_test)):.4f}")

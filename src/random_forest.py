"""
Random Forest Regressor Implementation
Exercise 2 - VU Machine Learning WS 2025

This module implements a Random Forest Regressor from scratch using:
- Bootstrap aggregating (bagging) of training samples
- Random feature subsampling at each split
- Ensemble averaging for predictions
- sklearn-compatible API (fit/predict/score)

Built on top of our optimized RegressionTree implementation.
The Random Forest reduces overfitting through ensemble averaging while
maintaining the interpretability benefits of decision trees.
"""

import numpy as np
from typing import Optional, Union, List
from regression_tree import RegressionTree


class RandomForestRegressor:
    """
    Random Forest for regression built on top of our RegressionTree.

    Uses bootstrap aggregating and random feature subsampling to create
    an ensemble of decorrelated decision trees. Final predictions are
    obtained by averaging predictions across all trees.

    Parameters:
        n_estimators: Number of trees in the forest (default=100)
        max_depth: Maximum depth for each tree (default=10)
        min_samples_split: Minimum samples to split an internal node (default=2)
        min_samples_leaf: Minimum samples required at a leaf node (default=1)
        max_features: Number of features to consider when looking for best split
                      - None: use all features
                      - int: exact number
                      - "sqrt": sqrt(n_features)
                      - "log2": log2(n_features)
                      (default="sqrt")
        bootstrap: Whether to use bootstrap sampling of rows (default=True)
        random_state: Seed for reproducibility (default=None)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[None, int, str] = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees_: List[RegressionTree] = []
        self.n_features_: Optional[int] = None

        # Initialize random number generator
        if random_state is not None:
            self._rng = np.random.RandomState(random_state)
        else:
            self._rng = np.random

    def _compute_max_features_int(self, n_features: int) -> int:
        """Convert max_features parameter to integer value."""
        if self.max_features is None:
            k = n_features
        elif isinstance(self.max_features, int):
            k = self.max_features
        elif self.max_features == "sqrt":
            k = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            k = int(np.log2(n_features))
        else:
            # Fallback: use all features
            k = n_features

        # Ensure k is in valid range [1, n_features]
        k = max(1, min(k, n_features))
        return k

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        """Create bootstrap sample by sampling rows with replacement."""
        n_samples = X.shape[0]
        indices = self._rng.randint(0, n_samples, size=n_samples)
        return X[indices], y[indices]

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestRegressor':
        """
        Fit the random forest to training data.

        Trains n_estimators trees on bootstrap samples (if bootstrap=True)
        with random feature subsampling at each split.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)

        Returns:
            self: Fitted RandomForestRegressor
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        max_features_int = self._compute_max_features_int(self.n_features_)

        self.trees_ = []

        # Train each tree in the forest
        for _ in range(self.n_estimators):
            # 1. Bootstrap sample of rows (if enabled)
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y

            # 2. Create and fit a tree with random feature subsampling
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features_int,
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values by averaging predictions over all trees.

        Args:
            X: Test features, shape (n_samples, n_features)

        Returns:
            predictions: Predicted values, shape (n_samples,)
        """
        X = np.asarray(X)
        # Collect predictions from each tree
        all_preds = np.array([tree.predict(X) for tree in self.trees_])
        # Average over trees (axis=0 is trees, axis=1 is samples)
        return np.mean(all_preds, axis=0)

    # ===== Utility Methods =====

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return R² score (sklearn-compatible).

        R² = 1 - (SS_res / SS_tot)
        where SS_res is the residual sum of squares and
        SS_tot is the total sum of squares.

        Args:
            X: Test features, shape (n_samples, n_features)
            y: True target values, shape (n_samples,)

        Returns:
            score: R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def get_avg_depth(self) -> float:
        """Return average depth across all trees in the forest."""
        if not self.trees_:
            return 0.0
        return np.mean([tree.get_depth() for tree in self.trees_])

    def get_avg_n_leaves(self) -> float:
        """Return average number of leaves across all trees."""
        if not self.trees_:
            return 0.0
        return np.mean([tree.get_n_leaves() for tree in self.trees_])

    def get_n_estimators(self) -> int:
        """Return number of trees actually fitted."""
        return len(self.trees_)


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    print("Random Forest Regressor Demo")
    print("=" * 50)

    # Generate synthetic regression data
    X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Single tree baseline
    print("\nBaseline: Single Regression Tree")
    print("-" * 50)
    single_tree = RegressionTree(max_depth=10)
    single_tree.fit(X_train, y_train)
    single_pred = single_tree.predict(X_test)
    print(f"Depth: {single_tree.get_depth()}")
    print(f"Leaves: {single_tree.get_n_leaves()}")
    print(f"Test R²: {r2_score(y_test, single_pred):.4f}")

    # Random Forest with default parameters
    print("\nRandom Forest (n_estimators=50, max_features='sqrt')")
    print("-" * 50)
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print(f"Number of trees: {rf.get_n_estimators()}")
    print(f"Avg depth: {rf.get_avg_depth():.1f}")
    print(f"Avg leaves: {rf.get_avg_n_leaves():.1f}")
    print(f"Test R²: {r2_score(y_test, rf_pred):.4f}")

    # Random Forest with more trees
    print("\nRandom Forest (n_estimators=100)")
    print("-" * 50)
    rf_large = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_large.fit(X_train, y_train)
    rf_large_pred = rf_large.predict(X_test)
    print(f"Number of trees: {rf_large.get_n_estimators()}")
    print(f"Avg depth: {rf_large.get_avg_depth():.1f}")
    print(f"Avg leaves: {rf_large.get_avg_n_leaves():.1f}")
    print(f"Test R²: {r2_score(y_test, rf_large_pred):.4f}")

    # Without bootstrap
    print("\nRandom Forest (bootstrap=False)")
    print("-" * 50)
    rf_no_bootstrap = RandomForestRegressor(
        n_estimators=50, max_depth=10, bootstrap=False, random_state=42
    )
    rf_no_bootstrap.fit(X_train, y_train)
    rf_no_bootstrap_pred = rf_no_bootstrap.predict(X_test)
    print(f"Test R²: {r2_score(y_test, rf_no_bootstrap_pred):.4f}")

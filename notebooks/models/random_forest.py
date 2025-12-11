
import numpy as np
from typing import Optional, Union, List
from .regression_tree import RegressionTree


class RandomForestRegressorScratch:
    """
    Random Forest for regression built on top of our RegressionTree.
    
    Parameters:
        n_estimators: number of trees in the forest
        max_depth: maximum depth for each tree
        min_samples_split: minimum samples to split an internal node
        min_samples_leaf: minimum samples required at a leaf node
        max_features: number of features to consider when looking for best split.
                      Can be:
                        - None  -> use all features
                        - int   -> exact number
                        - "sqrt" -> sqrt(n_features)
                        - "log2" -> log2(n_features)
        bootstrap: whether to use bootstrap sampling of rows
        random_state: seed for reproducibility
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

        if random_state is not None:
            self._rng = np.random.RandomState(random_state)
        else:
            self._rng = np.random

    def _compute_max_features_int(self, n_features: int) -> int:
        """Convert self.max_features (None/int/str) to an integer value."""
        if self.max_features is None:
            k = n_features
        elif isinstance(self.max_features, int):
            k = self.max_features
        elif self.max_features == "sqrt":
            k = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            k = int(np.log2(n_features))
        else:
            # fallback: all features
            k = n_features

        k = max(1, min(k, n_features))
        return k

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        """Sample rows with replacement to create a bootstrap sample."""
        n_samples = X.shape[0]
        indices = self._rng.randint(0, n_samples, size=n_samples)
        return X[indices], y[indices]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressorScratch":
        """Fit the random forest to training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        max_features_int = self._compute_max_features_int(self.n_features_)

        self.trees_ = []

        for _ in range(self.n_estimators):
            # 1. Bootstrap sample of rows
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y

            # 2. Create and fit a tree
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features_int,   # integer, tree will randomly choose this many features at each split
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by averaging predictions over all trees."""
        X = np.asarray(X)
        # collect predictions from each tree
        all_preds = np.array([tree.predict(X) for tree in self.trees_])  # shape: (n_trees, n_samples)
        # average over trees
        return np.mean(all_preds, axis=0)

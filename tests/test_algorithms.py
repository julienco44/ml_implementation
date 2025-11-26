"""
Unit Tests for Regression Algorithms
Exercise 2 - VU Machine Learning WS 2025

Run with: pytest tests/test_algorithms.py
"""

import numpy as np
import pytest
import sys
sys.path.append('../src')

from regression_tree import RegressionTree, Node
from random_forest import RandomForest


class TestRegressionTree:
    """Test cases for RegressionTree implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
        return X, y
    
    def test_tree_initialization(self):
        """Test that tree initializes correctly."""
        tree = RegressionTree(max_depth=5, min_samples_split=10)
        assert tree.max_depth == 5
        assert tree.min_samples_split == 10
        assert tree.root is None
    
    def test_tree_fit(self, sample_data):
        """Test that tree can be fitted."""
        X, y = sample_data
        tree = RegressionTree(max_depth=5)
        tree.fit(X, y)
        assert tree.root is not None
        assert hasattr(tree, 'n_features_')
    
    def test_tree_predict_shape(self, sample_data):
        """Test that predictions have correct shape."""
        X, y = sample_data
        tree = RegressionTree(max_depth=5)
        tree.fit(X, y)
        predictions = tree.predict(X)
        assert predictions.shape == y.shape
    
    def test_tree_predict_values(self, sample_data):
        """Test that predictions are reasonable."""
        X, y = sample_data
        tree = RegressionTree(max_depth=5)
        tree.fit(X, y)
        predictions = tree.predict(X)
        # Predictions should not be NaN or Inf
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))
        # Predictions should be in a reasonable range
        assert np.all(predictions >= y.min() - 10)
        assert np.all(predictions <= y.max() + 10)
    
    def test_single_sample_prediction(self, sample_data):
        """Test prediction on a single sample."""
        X, y = sample_data
        tree = RegressionTree(max_depth=5)
        tree.fit(X, y)
        single_pred = tree.predict(X[:1])
        assert single_pred.shape == (1,)


class TestRandomForest:
    """Test cases for RandomForest implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
        return X, y
    
    def test_forest_initialization(self):
        """Test that forest initializes correctly."""
        rf = RandomForest(n_trees=10, max_depth=5)
        assert rf.n_trees == 10
        assert rf.max_depth == 5
        assert len(rf.trees) == 0
    
    def test_forest_fit(self, sample_data):
        """Test that forest can be fitted."""
        X, y = sample_data
        rf = RandomForest(n_trees=10, max_depth=5)
        rf.fit(X, y)
        assert len(rf.trees) == 10
        # All trees should be fitted
        for tree in rf.trees:
            assert tree.root is not None
    
    def test_forest_predict_shape(self, sample_data):
        """Test that predictions have correct shape."""
        X, y = sample_data
        rf = RandomForest(n_trees=10, max_depth=5)
        rf.fit(X, y)
        predictions = rf.predict(X)
        assert predictions.shape == y.shape
    
    def test_forest_predict_values(self, sample_data):
        """Test that predictions are reasonable."""
        X, y = sample_data
        rf = RandomForest(n_trees=10, max_depth=5)
        rf.fit(X, y)
        predictions = rf.predict(X)
        # Predictions should not be NaN or Inf
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))
        # Predictions should be in a reasonable range
        assert np.all(predictions >= y.min() - 10)
        assert np.all(predictions <= y.max() + 10)
    
    def test_forest_better_than_single_tree(self, sample_data):
        """Test that forest generally performs better than single tree."""
        X, y = sample_data
        
        # Train single tree
        tree = RegressionTree(max_depth=5)
        tree.fit(X, y)
        tree_predictions = tree.predict(X)
        tree_mse = np.mean((y - tree_predictions) ** 2)
        
        # Train forest
        rf = RandomForest(n_trees=50, max_depth=5, random_state=42)
        rf.fit(X, y)
        rf_predictions = rf.predict(X)
        rf_mse = np.mean((y - rf_predictions) ** 2)
        
        # Forest should generally have lower or similar MSE
        # Note: This might not always be true due to randomness
        assert rf_mse <= tree_mse * 1.5  # Allow some margin
    
    def test_forest_reproducibility(self, sample_data):
        """Test that forest produces same results with same random state."""
        X, y = sample_data
        
        rf1 = RandomForest(n_trees=10, random_state=42)
        rf1.fit(X, y)
        pred1 = rf1.predict(X)
        
        rf2 = RandomForest(n_trees=10, random_state=42)
        rf2.fit(X, y)
        pred2 = rf2.predict(X)
        
        np.testing.assert_array_almost_equal(pred1, pred2)


class TestNode:
    """Test cases for Node class."""
    
    def test_leaf_node(self):
        """Test leaf node creation."""
        node = Node(value=5.0)
        assert node.is_leaf()
        assert node.value == 5.0
        assert node.feature_idx is None
        assert node.threshold is None
    
    def test_internal_node(self):
        """Test internal node creation."""
        left = Node(value=3.0)
        right = Node(value=7.0)
        node = Node(feature_idx=2, threshold=5.0, left=left, right=right)
        
        assert not node.is_leaf()
        assert node.feature_idx == 2
        assert node.threshold == 5.0
        assert node.left == left
        assert node.right == right


def test_integration(sample_data):
    """Integration test for full pipeline."""
    X, y = sample_data
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train models
    tree = RegressionTree(max_depth=5)
    tree.fit(X_train, y_train)
    
    rf = RandomForest(n_trees=20, max_depth=5)
    rf.fit(X_train, y_train)
    
    # Make predictions
    tree_pred = tree.predict(X_test)
    rf_pred = rf.predict(X_test)
    
    # Check predictions are valid
    assert tree_pred.shape == y_test.shape
    assert rf_pred.shape == y_test.shape
    assert not np.any(np.isnan(tree_pred))
    assert not np.any(np.isnan(rf_pred))


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
    return X, y


if __name__ == "__main__":
    # Run tests with: python tests/test_algorithms.py
    pytest.main([__file__, "-v"])

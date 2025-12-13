# Machine Learning Exercise 2 - Regression Trees and Random Forests

## 📋 Project Overview

This repository contains the implementation of regression tree and random forest algorithms for the Machine Learning course Exercise 2.

## 👥 Team Members

- Student 1: Julian Hardt - 12330562
- Student 2: [Name] - [Matriculation Number]
- Student 3: [Name] - [Matriculation Number]

## 📁 Repository Structure

```
ml_implementation/
├── src/                        
│   ├── regression_tree.py      # Regression tree with CCP (from scratch)
│   └── utils.py                
├── notebooks/                  
│   ├── regression_tree_analysis.ipynb  # Main analysis notebook
│   ├── preprocessing_ames_housing.ipynb
│   └── preprocessing_student_perfomance.ipynb
├── data/processed/             # Preprocessed .npy files
└── results/                    # Output files
```

## 🚀 How to Use the Regression Tree

### Option 1: Use `regression_tree_analysis.ipynb`

1. Preprocess your dataset and save as `.npy` files in `data/processed/`
2. Open `notebooks/regression_tree_analysis.ipynb`
3. Change the data loading path to your dataset
4. Run all cells

### Option 2: Use `src/regression_tree.py` directly

```python
import sys
sys.path.insert(0, 'src')
from regression_tree import RegressionTree

# Train
tree = RegressionTree(max_depth=10)
tree.fit(X_train, y_train)

# Predict
predictions = tree.predict(X_test)

# With Cost-Complexity Pruning
tree_pruned = RegressionTree(max_depth=15, ccp_alpha=1e8)
tree_pruned.fit(X_train, y_train)
```

### Available Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | None | Maximum tree depth |
| `min_samples_split` | 2 | Min samples to split a node |
| `min_samples_leaf` | 1 | Min samples in a leaf |
| `max_features` | None | Number of features to consider |
| `ccp_alpha` | 0.0 | Pruning parameter (higher = more pruning) |

## 📅 Important Dates

- **Submission Deadline**: 15.12.2025, 14:00

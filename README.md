# Machine Learning Exercise 2 - Regression Trees and Random Forests


## 📋 Project Overview

This repository contains the implementation of regression tree and random forest algorithms for the Machine Learning course Exercise 2.

## 👥 Team Members

- Student 1: [Name] - [Matriculation Number]
- Student 2: [Name] - [Matriculation Number]
- Student 3: [Name] - [Matriculation Number]

## 🎯 Assignment Requirements

### Algorithms to Implement
1. **Regression Tree Algorithm** - Implemented from scratch
2. **Random Forest Algorithm** - Built on top of the regression tree implementation

### Datasets
3 regression datasets with different characteristics:
1. Dataset 1: [Name] - [Brief description, samples, dimensions]
2. Dataset 2: [Name] - [Brief description, samples, dimensions]
3. Dataset 3: [Name] - [Brief description, samples, dimensions]

### Evaluation
- Compare with existing implementations (scikit-learn, etc.)
- Test at least 3 different configurations for Random Forest
- Use at least 2 performance metrics
- Apply cross-validation

## 📁 Repository Structure

```
ml-exercise2/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── data/                       # Dataset links and descriptions
│   └── datasets.md
├── src/                        # Source code
│   ├── regression_tree.py      # Regression tree implementation
│   ├── random_forest.py        # Random forest implementation
│   ├── utils.py                # Utility functions
│   └── experiments.py          # Experimental setup and evaluation
├── notebooks/                  # Jupyter notebooks (optional)
│   └── exploration.ipynb
├── results/                    # Experimental results
│   ├── figures/                # Plots and visualizations
│   └── metrics/                # Performance metrics
├── slides/                     # Presentation slides
│   └── presentation.pptx
└── tests/                      # Unit tests (optional)
    └── test_algorithms.py
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-exercise2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Code

```bash
# Run experiments
python src/experiments.py

# Or use individual algorithms
python src/regression_tree.py
python src/random_forest.py
```

## 📊 Performance Metrics

- Mean Squared Error (MSE)
- R² Score
- [Add other metrics used]

## 🔬 Experimental Results

[To be updated after running experiments]

### Dataset 1: [Name]
| Algorithm | Configuration | MSE | R² | Time (s) |
|-----------|---------------|-----|-----|----------|
| Our Regression Tree | - | - | - | - |
| Our Random Forest | n_trees=10 | - | - | - |
| Our Random Forest | n_trees=50 | - | - | - |
| Our Random Forest | n_trees=100 | - | - | - |
| Sklearn Random Forest | default | - | - | - |
| [Other baseline] | default | - | - | - |

## 📝 Key Findings

[To be updated after experiments]

- Finding 1: ...
- Finding 2: ...
- Finding 3: ...


## 📅 Important Dates

- **Submission Deadline**: 15.12.2025, 14:00


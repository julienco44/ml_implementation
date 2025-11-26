# Setup Guide

This guide will help you set up the project and get started with Exercise 2.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

## Initial Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd ml-exercise2
```

### 2. Create a Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
ml-exercise2/
├── README.md              # Project overview and documentation
├── SETUP.md              # This file
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
│
├── data/                # Datasets directory
│   └── datasets.md      # Dataset documentation
│
├── src/                 # Source code
│   ├── regression_tree.py    # Regression tree implementation
│   ├── random_forest.py      # Random forest implementation
│   ├── utils.py             # Utility functions
│   └── experiments.py        # Main experiments script
│
├── notebooks/           # Jupyter notebooks
│   └── exploration.ipynb    # Data exploration notebook
│
├── results/            # Results directory
│   ├── figures/        # Plots and visualizations
│   └── metrics/        # Performance metrics
│
├── slides/             # Presentation slides
│   └── presentation.pptx
│
└── tests/              # Unit tests (optional)
    └── test_algorithms.py
```

## Getting Started

### Step 1: Find and Download Datasets

1. Visit the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/
2. Or Kaggle: https://www.kaggle.com/datasets
3. Find 3 regression datasets published after 2020
4. Download and place them in the `data/` directory
5. Update `data/datasets.md` with dataset information

### Step 2: Implement the Algorithms

Start with the regression tree:
```bash
python src/regression_tree.py
```

The file contains:
- A `Node` class for tree structure
- A `RegressionTree` class with:
  - `fit()` method to build the tree
  - `predict()` method to make predictions
  - Helper methods marked with TODO

Complete the TODO sections to implement:
- Tree building logic
- Split finding algorithm
- Prediction logic

### Step 3: Implement Random Forest

After completing the regression tree:
```bash
python src/random_forest.py
```

Build the random forest by:
- Using your regression tree implementation
- Implementing bootstrap sampling
- Aggregating predictions from multiple trees

### Step 4: Run Experiments

Once both algorithms are implemented:
```bash
python src/experiments.py
```

This will:
- Load all datasets
- Train all models
- Perform cross-validation
- Compare results
- Save metrics to `results/metrics/`

### Step 5: Explore Data and Results

Use Jupyter notebook for exploration:
```bash
jupyter notebook notebooks/exploration.ipynb
```

This notebook helps you:
- Explore dataset characteristics
- Visualize model performance
- Create plots for your presentation

### Step 6: Create Presentation

1. Create slides in the `slides/` directory
2. Include:
   - Dataset characteristics
   - Implementation details
   - Experimental results
   - Comparisons with baselines
   - Conclusions

## Development Workflow

### Testing Your Implementation

1. **Unit Testing**: Test individual components
   ```python
   # Test regression tree
   python src/regression_tree.py
   
   # Test random forest
   python src/random_forest.py
   ```

2. **Cross-Validation**: Use the utils module
   ```python
   from utils import cross_validate
   cv_results = cross_validate(model, X, y, n_splits=5)
   ```

3. **Compare with Sklearn**: Ensure your implementation is correct
   ```python
   from sklearn.ensemble import RandomForestRegressor
   sklearn_rf = RandomForestRegressor(n_estimators=100)
   ```

### Adding New Features

To add a new feature or experiment:

1. Create a new branch:
   ```bash
   git checkout -b feature/new-experiment
   ```

2. Make your changes

3. Test thoroughly

4. Commit and push:
   ```bash
   git add .
   git commit -m "Add new experiment"
   git push origin feature/new-experiment
   ```

## Common Issues and Solutions

### Issue: Module not found
**Solution**: Make sure you're in the correct directory and virtual environment is activated.

### Issue: Dataset loading errors
**Solution**: Check file paths and formats in `src/utils.py`

### Issue: Memory errors with large datasets
**Solution**: Consider using data sampling or reducing the number of trees

### Issue: Slow training
**Solution**: 
- Reduce `max_depth`
- Reduce `n_trees` for random forest
- Use smaller datasets for testing

## Performance Optimization Tips

1. **Use NumPy vectorization** instead of loops where possible
2. **Profile your code** to find bottlenecks:
   ```python
   import cProfile
   cProfile.run('your_function()')
   ```
3. **Consider parallel processing** for random forest training

## Assignment Checklist

Before submission, ensure you have:

- [ ] Implemented regression tree from scratch
- [ ] Implemented random forest from scratch
- [ ] Selected 3 diverse datasets (published after 2020)
- [ ] Tested at least 3 random forest configurations
- [ ] Compared with existing implementations
- [ ] Used at least 2 performance metrics
- [ ] Applied cross-validation
- [ ] Created 20-40 slides with results
- [ ] Documented dataset sources in `data/datasets.md`
- [ ] Included all source code
- [ ] Prepared for 15-minute discussion

## Submission

Create a zip file containing:
```
ml-exercise2-submission.zip
├── src/                    # All source code
├── data/datasets.md        # Dataset links and info
├── slides/                 # Presentation
├── requirements.txt        # Dependencies
└── README.md              # Project overview
```

**Deadline**: 15.12.2025, 14:00

## Discussion Preparation

Be ready to explain:
- Your implementation approach
- Splitting criteria used
- Why you chose specific hyperparameters
- Performance differences between algorithms
- Challenges faced and solutions
- Key findings and insights

## Resources

- Course materials on TUWEL
- Scikit-learn documentation: https://scikit-learn.org/
- NumPy documentation: https://numpy.org/
- Decision Trees theory: Chapter on Trees in your textbook

## Getting Help

If you encounter issues:
1. Check this documentation
2. Review course materials
3. Consult with team members
4. Post in the course forum
5. Attend office hours

## License

This project is for educational purposes as part of VU Machine Learning at TU Wien.

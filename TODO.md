# TODO List for Exercise 2

## Phase 1: Dataset Selection and Preparation (Week 1)

### Dataset Tasks
- [ ] Find Dataset 1 from UCI ML Repository or Kaggle (published after 2020)
  - [ ] Download dataset
  - [ ] Document in `data/datasets.md`
  - [ ] Verify it has different characteristics (small samples, low dimensions)
  
- [ ] Find Dataset 2 from UCI ML Repository or Kaggle (published after 2020)
  - [ ] Download dataset
  - [ ] Document in `data/datasets.md`
  - [ ] Verify it has different characteristics (medium samples, medium dimensions)
  
- [ ] Find Dataset 3 from UCI ML Repository or Kaggle (published after 2020)
  - [ ] Download dataset
  - [ ] Document in `data/datasets.md`
  - [ ] Verify it has different characteristics (large samples, high dimensions)

### Data Preprocessing
- [ ] Implement data loading in `utils.py`
- [ ] Handle missing values
- [ ] Normalize/standardize features if needed
- [ ] Split data for training and testing

## Phase 2: Regression Tree Implementation (Week 1-2)

### Core Implementation (`src/regression_tree.py`)
- [ ] Complete `_build_tree()` method
  - [ ] Implement stopping criteria (max_depth, min_samples_split)
  - [ ] Handle leaf node creation
  - [ ] Implement recursive splitting
  
- [ ] Complete `_find_best_split()` method
  - [ ] Iterate over all features
  - [ ] Try different split thresholds
  - [ ] Calculate split quality (variance reduction or MSE)
  - [ ] Return best feature and threshold
  
- [ ] Complete `_calculate_split_score()` method
  - [ ] Implement weighted MSE calculation
  - [ ] Or implement variance reduction
  
- [ ] Test regression tree
  - [ ] Test on simple synthetic data
  - [ ] Verify predictions make sense
  - [ ] Check tree structure

### Advanced Features (Optional)
- [ ] Implement pruning
- [ ] Add support for categorical features
- [ ] Implement feature importance calculation

## Phase 3: Random Forest Implementation (Week 2)

### Core Implementation (`src/random_forest.py`)
- [ ] Complete `fit()` method
  - [ ] Implement bootstrap sampling
  - [ ] Set max_features (sqrt or n_features/3)
  - [ ] Train multiple regression trees
  - [ ] Store all trees
  
- [ ] Complete `predict()` method
  - [ ] Get predictions from all trees
  - [ ] Average predictions
  
- [ ] Test random forest
  - [ ] Test with different number of trees (10, 50, 100)
  - [ ] Compare with single tree
  - [ ] Verify ensemble effect

### Advanced Features (Optional)
- [ ] Implement `get_feature_importance()`
- [ ] Add out-of-bag (OOB) error estimation
- [ ] Implement parallel tree training

## Phase 4: Experiments and Evaluation (Week 2-3)

### Experimental Setup (`src/experiments.py`)
- [ ] Update dataset paths
- [ ] Configure regression tree hyperparameters
- [ ] Configure random forest configurations:
  - [ ] Config 1: 10 trees
  - [ ] Config 2: 50 trees
  - [ ] Config 3: 100 trees
  - [ ] Consider testing different max_depth values
  
### Baseline Comparisons
- [ ] Compare with sklearn DecisionTreeRegressor
- [ ] Compare with sklearn RandomForestRegressor
- [ ] Compare with at least one other technique (e.g., LinearRegression, SVR, GradientBoosting)

### Evaluation Metrics
- [ ] Calculate MSE (Mean Squared Error)
- [ ] Calculate R² Score
- [ ] Consider adding MAE, RMSE
- [ ] Implement cross-validation (5-fold or 10-fold)

### Results Analysis
- [ ] Record training time for each algorithm
- [ ] Record prediction time
- [ ] Compare accuracy across datasets
- [ ] Analyze where each algorithm performs best/worst

## Phase 5: Visualization and Documentation (Week 3)

### Create Visualizations
- [ ] Plot predicted vs actual values
- [ ] Create residual plots
- [ ] Visualize model comparison (bar charts)
- [ ] Create learning curves (optional)
- [ ] Plot feature importance (if implemented)
- [ ] Save all figures to `results/figures/`

### Generate Result Tables
- [ ] Create comparison tables for all models
- [ ] Include mean and std dev for CV results
- [ ] Format tables for presentation
- [ ] Save results to `results/metrics/`

### Code Documentation
- [ ] Add docstrings to all functions
- [ ] Add comments for complex logic
- [ ] Update README with final results
- [ ] Ensure code follows style guidelines

## Phase 6: Presentation Creation (Week 3-4)

### Slide Content (20-40 slides)
- [ ] Title slide (1 slide)
- [ ] Overview/Introduction (2-3 slides)
- [ ] Dataset descriptions (3-4 slides)
  - [ ] Dataset 1 characteristics
  - [ ] Dataset 2 characteristics
  - [ ] Dataset 3 characteristics
  
- [ ] Implementation details (5-7 slides)
  - [ ] Regression tree algorithm
  - [ ] Pseudocode or flowchart
  - [ ] Random forest algorithm
  - [ ] Key implementation decisions
  - [ ] Challenges faced and solutions
  
- [ ] Experimental setup (2-3 slides)
  - [ ] Hyperparameters tested
  - [ ] Evaluation methodology
  - [ ] Cross-validation approach
  
- [ ] Results (8-12 slides)
  - [ ] Dataset 1 results (table + figures)
  - [ ] Dataset 2 results (table + figures)
  - [ ] Dataset 3 results (table + figures)
  - [ ] Overall comparison
  
- [ ] Discussion and Analysis (3-5 slides)
  - [ ] Performance comparison
  - [ ] Efficiency analysis
  - [ ] When does each algorithm work best?
  - [ ] Unexpected findings
  
- [ ] Conclusions (2-3 slides)
  - [ ] Key findings
  - [ ] Lessons learned
  - [ ] Future improvements
  
- [ ] References (1 slide)

### Presentation Preparation
- [ ] Ensure all team members understand all parts
- [ ] Prepare to answer questions about:
  - [ ] Implementation choices
  - [ ] Splitting criteria
  - [ ] Performance metrics
  - [ ] Any part of the code
- [ ] Practice 15-minute presentation
- [ ] Prepare for potential questions

## Phase 7: Final Submission (Week 4)

### Pre-submission Checklist
- [ ] Run all code one final time to ensure it works
- [ ] Verify all datasets are documented
- [ ] Check all results are saved
- [ ] Review presentation slides
- [ ] Update README with final information
- [ ] Check requirements.txt is complete

### Create Submission Package
- [ ] Create zip file with:
  - [ ] Source code (`src/` folder)
  - [ ] Dataset documentation (`data/datasets.md`)
  - [ ] Presentation slides (`slides/` folder)
  - [ ] Requirements file
  - [ ] README
  
### Submit
- [ ] Submit before deadline: **15.12.2025, 14:00**
- [ ] Verify submission was successful
- [ ] Keep a backup copy

### Discussion Preparation
- [ ] Schedule discussion slot (16.12-19.12 or 8.1-9.1)
- [ ] Review all code
- [ ] Review all slides
- [ ] Practice explanations
- [ ] Each team member prepares to answer questions

## Team Coordination

### Responsibilities (Assign to team members)
- [ ] Person 1: Regression tree implementation + Dataset 1
- [ ] Person 2: Random forest implementation + Dataset 2
- [ ] Person 3: Experiments & evaluation + Dataset 3
- [ ] All: Presentation preparation and practice

### Regular Check-ins
- [ ] Week 1: Dataset selection complete
- [ ] Week 2: Regression tree working
- [ ] Week 2.5: Random forest working
- [ ] Week 3: Experiments complete
- [ ] Week 3.5: Slides draft ready
- [ ] Week 4: Final review and submission

## Notes

### Key Requirements from Assignment
- ✅ Groups of 3 students
- ✅ Implement regression tree from scratch
- ✅ Implement random forest from scratch
- ✅ 3 datasets with different characteristics
- ✅ Compare with existing implementations
- ✅ At least 3 random forest configurations
- ✅ At least 2 performance metrics
- ✅ Apply cross-validation
- ✅ 20-40 slides
- ✅ No report needed
- ✅ Submit by 15.12.2025, 14:00
- ✅ 15-minute discussion

### Important Reminders
- Do NOT use existing code for the core algorithms
- You CAN use sklearn for comparison
- You CAN use utility functions (cross-validation, metrics, etc.)
- Document everything clearly
- Every team member should understand all parts

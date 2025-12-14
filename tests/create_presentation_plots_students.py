"""
Create Presentation Visualizations for Student Performance Dataset
Run: python create_presentation_plots_students.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from regression_tree import RegressionTree
from random_forest import RandomForestRegressor
from sklearn.metrics import r2_score

# Set professional style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'font.family': 'serif',
})

OUTPUT_DIR = Path(__file__).parent / 'results' / 'figures_students'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent / 'data' / 'processed'

COLORS = {
    'train': '#2c3e50',
    'test': '#e67e22',
    'our_rf': '#27ae60',
}


def load_data():
    """Load Student Performance dataset."""
    X_train = np.load(DATA_DIR / 'students_X_train.npy')
    X_test = np.load(DATA_DIR / 'students_X_test.npy')
    y_train = np.load(DATA_DIR / 'students_y_train.npy')
    y_test = np.load(DATA_DIR / 'students_y_test.npy')

    print(f"Student Performance Dataset loaded:")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def run_n_estimators_experiment(X_train, X_test, y_train, y_test):
    """Run n_estimators hyperparameter experiment."""
    print("\nRunning n_estimators experiment...")
    n_estimators_values = [10, 20, 30, 50, 75, 100, 150]
    train_r2_list = []
    test_r2_list = []

    for n_est in n_estimators_values:
        print(f"  n_estimators={n_est}...", end=" ")
        rf = RandomForestRegressor(n_estimators=n_est, max_depth=10, min_samples_split=10,
                                   min_samples_leaf=5, max_features='sqrt', random_state=42)
        rf.fit(X_train, y_train)
        train_r2 = rf.score(X_train, y_train)
        test_r2 = rf.score(X_test, y_test)
        train_r2_list.append(train_r2)
        test_r2_list.append(test_r2)
        print(f"Test R²={test_r2:.4f}")

    return n_estimators_values, train_r2_list, test_r2_list


def run_max_depth_experiment(X_train, X_test, y_train, y_test):
    """Run max_depth hyperparameter experiment."""
    print("\nRunning max_depth experiment...")
    max_depth_values = [3, 5, 8, 10, 12, 15, 20]
    train_r2_list = []
    test_r2_list = []

    for depth in max_depth_values:
        print(f"  max_depth={depth}...", end=" ")
        rf = RandomForestRegressor(n_estimators=30, max_depth=depth, min_samples_split=10,
                                   min_samples_leaf=5, max_features='sqrt', random_state=42)
        rf.fit(X_train, y_train)
        train_r2 = rf.score(X_train, y_train)
        test_r2 = rf.score(X_test, y_test)
        train_r2_list.append(train_r2)
        test_r2_list.append(test_r2)
        print(f"Test R²={test_r2:.4f}")

    return max_depth_values, train_r2_list, test_r2_list


def run_ccp_experiment(X_train, X_test, y_train, y_test):
    """Run CCP (Cost-Complexity Pruning) experiment."""
    print("\nRunning CCP experiment...")

    # First get the CCP path
    tree_for_path = RegressionTree(max_depth=20)
    path = tree_for_path.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas_full = path['ccp_alphas']

    # Sample alphas across the range
    if len(ccp_alphas_full) > 10:
        indices = np.linspace(0, len(ccp_alphas_full)-1, 10, dtype=int)
        ccp_alphas = ccp_alphas_full[indices]
    else:
        ccp_alphas = ccp_alphas_full

    test_r2_list = []
    n_leaves_list = []

    for alpha in ccp_alphas:
        print(f"  ccp_alpha={alpha:.2e}...", end=" ")
        tree = RegressionTree(max_depth=15, ccp_alpha=alpha)
        tree.fit(X_train, y_train)
        test_r2 = tree.score(X_test, y_test)
        n_leaves = tree.get_n_leaves()
        test_r2_list.append(test_r2)
        n_leaves_list.append(n_leaves)
        print(f"Test R²={test_r2:.4f}, Leaves={n_leaves}")

    return list(ccp_alphas), test_r2_list, n_leaves_list


def create_n_estimators_impact(n_estimators_values, train_r2, test_r2):
    """Create n_estimators impact plot."""
    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(n_estimators_values, train_r2, 'o--', label='Train R²',
            color=COLORS['train'], linewidth=2.5, markersize=9, alpha=0.7)
    ax.plot(n_estimators_values, test_r2, 's-', label='Test R²',
            color=COLORS['test'], linewidth=2.5, markersize=9)

    optimal_idx = test_r2.index(max(test_r2))
    optimal_n = n_estimators_values[optimal_idx]
    ax.axvline(x=optimal_n, color=COLORS['our_rf'], linestyle=':', linewidth=2.5, label=f'Optimal: {optimal_n}')
    ax.plot(optimal_n, test_r2[optimal_idx], 'g*', markersize=20)

    ax.set_xlabel('Number of Trees (n_estimators)', fontweight='bold')
    ax.set_ylabel('R² Score', fontweight='bold')
    ax.set_title('Impact of Ensemble Size (Student Performance)', fontweight='bold', pad=20)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide_c_n_estimators_impact.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'slide_c_n_estimators_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: Slide C - n_estimators impact")


def create_max_depth_impact(max_depth_values, train_r2, test_r2):
    """Create max_depth impact plot."""
    fig, ax1 = plt.subplots(figsize=(11, 7))

    ax1.plot(max_depth_values, train_r2, 'o--', label='Train R²',
            color=COLORS['train'], linewidth=2.5, markersize=9)
    ax1.plot(max_depth_values, test_r2, 's-', label='Test R²',
            color=COLORS['test'], linewidth=2.5, markersize=9)
    ax1.set_xlabel('Maximum Tree Depth', fontweight='bold')
    ax1.set_ylabel('R² Score', fontweight='bold')
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    gaps = [t - te for t, te in zip(train_r2, test_r2)]
    ax2.plot(max_depth_values, gaps, '^-', label='Overfitting Gap', color='#e74c3c', linewidth=2.5, markersize=8)
    ax2.set_ylabel('Overfitting Gap', color='#e74c3c', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')

    ax1.set_title('Bias-Variance Trade-off: max_depth (Student Performance)', fontweight='bold', pad=20)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide_d_max_depth_impact.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'slide_d_max_depth_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: Slide D - max_depth impact")


def create_ccp_tradeoff(ccp_alphas, test_r2, n_leaves):
    """Create CCP tradeoff plot."""
    fig, ax1 = plt.subplots(figsize=(12, 7))

    color1 = COLORS['test']
    ax1.set_xlabel('CCP Alpha (log scale)', fontweight='bold')
    ax1.set_ylabel('Test R²', color=color1, fontweight='bold')

    # Handle alpha=0 for log scale
    alphas_plot = [max(a, 1e-6) for a in ccp_alphas]
    line1, = ax1.semilogx(alphas_plot, test_r2, 'o-', color=color1, linewidth=3, markersize=10, label='Test R²')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    color2 = '#9b59b6'
    ax2.set_ylabel('Number of Leaves', color=color2, fontweight='bold')
    line2, = ax2.semilogx(alphas_plot, n_leaves, 's-', color=color2, linewidth=3, markersize=10, label='Tree Size')
    ax2.tick_params(axis='y', labelcolor=color2)

    optimal_idx = test_r2.index(max(test_r2))
    ax1.plot(alphas_plot[optimal_idx], test_r2[optimal_idx], 'g*', markersize=25, zorder=10, label='Optimal α')

    ax1.set_title('Cost-Complexity Pruning Trade-off (Student Performance)', fontweight='bold', pad=20)
    lines = [line1, line2]
    labels = ['Test R²', 'Tree Size']
    ax1.legend(lines, labels, loc='center right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide_f_ccp_tradeoff.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'slide_f_ccp_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: Slide F - CCP tradeoff")


if __name__ == "__main__":
    print("=" * 60)
    print("Creating Presentation Visualizations - Student Performance")
    print("=" * 60)

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Run experiments
    n_est_vals, n_est_train, n_est_test = run_n_estimators_experiment(X_train, X_test, y_train, y_test)
    depth_vals, depth_train, depth_test = run_max_depth_experiment(X_train, X_test, y_train, y_test)
    ccp_alphas, ccp_test, ccp_leaves = run_ccp_experiment(X_train, X_test, y_train, y_test)

    # Create plots
    print("\nCreating plots...")
    create_n_estimators_impact(n_est_vals, n_est_train, n_est_test)
    create_max_depth_impact(depth_vals, depth_train, depth_test)
    create_ccp_tradeoff(ccp_alphas, ccp_test, ccp_leaves)

    print()
    print(f"All plots saved to: {OUTPUT_DIR.resolve()}")
    print("  - PDF format (for presentations)")
    print("  - PNG format (for backup)")

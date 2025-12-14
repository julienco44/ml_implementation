"""
Create Presentation Visualizations for ML Exercise 2
Run: python create_presentation_plots.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'font.family': 'serif',
})

OUTPUT_DIR = Path('results/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'linear': '#95a5a6',
    'single_tree': '#e74c3c',
    'our_rf': '#27ae60',
    'sklearn_rf': '#3498db',
    'train': '#2c3e50',
    'test': '#e67e22',
}

# ============ ÄNDERE DIESE WERTE ============
datasets = ['Ames Housing', 'Student Perf', 'IOT Temp']

# Test R² scores
linear_reg_scores = [-50.28, 0.141, 0.178]
single_tree_scores = [0.763, -0.243, 0.897]
our_rf_scores = [0.820, 0.228, 0.837]
sklearn_rf_scores = [0.811, 0.127, 0.844]

# Overfitting gaps
single_tree_gaps = [0.157, 0.864, 0.050]
our_rf_gaps = [0.072, 0.312, -0.032]

# Hyperparameter experiments
n_estimators_values = [10, 20, 30, 50, 75, 100, 150]
n_estimators_train_r2 = [0.920, 0.920, 0.920, 0.920, 0.919, 0.919, 0.918]
n_estimators_test_r2 = [0.78, 0.80, 0.81, 0.820, 0.823, 0.824, 0.824]

max_depth_values = [3, 5, 8, 10, 12, 15, 20]
max_depth_train_r2 = [0.82, 0.89, 0.91, 0.92, 0.93, 0.94, 0.95]
max_depth_test_r2 = [0.79, 0.81, 0.82, 0.824, 0.827, 0.820, 0.815]

# CCP experiments
ccp_alphas = [0, 1e6, 5e7, 1e8, 2e8, 5e8, 1e9]
ccp_test_r2 = [0.743, 0.744, 0.746, 0.747, 0.747, 0.745, 0.730]
ccp_n_leaves = [760, 650, 450, 320, 259, 180, 85]
# ============================================


def create_performance_comparison():
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(datasets))
    width = 0.2
    
    # Clip extreme values for visualization
    linear_clipped = [max(v, -1.2) for v in linear_reg_scores]
    single_clipped = [max(v, -1.2) for v in single_tree_scores]
    
    ax.bar(x - 1.5*width, linear_clipped, width, label='Linear Regression', color=COLORS['linear'], alpha=0.85)
    ax.bar(x - 0.5*width, single_clipped, width, label='Single Tree (Ours)', color=COLORS['single_tree'], alpha=0.85)
    ax.bar(x + 0.5*width, our_rf_scores, width, label='Random Forest (Ours)', color=COLORS['our_rf'], alpha=0.85)
    ax.bar(x + 1.5*width, sklearn_rf_scores, width, label='sklearn RF', color=COLORS['sklearn_rf'], alpha=0.85)
    
    ax.set_ylabel('Test R² Score', fontweight='bold')
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_title('Model Performance Comparison', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=True)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    # Limit Y-axis to show meaningful range
    ax.set_ylim([-1.5, 1.1])
    
    # Annotate extreme/truncated values
    for i, val in enumerate(linear_reg_scores):
        if val < -1.2:
            ax.text(i - 1.5*width, -1.35, f'{val:.1f}', ha='center', fontsize=9, 
                   style='italic', color='gray', rotation=0)
            ax.annotate('', xy=(i - 1.5*width, -1.2), xytext=(i - 1.5*width, -1.28),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    
    for i, val in enumerate(single_tree_scores):
        if val < -1.2:
            ax.text(i - 0.5*width, -1.35, f'{val:.2f}', ha='center', fontsize=9,
                   style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide_a_performance_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'slide_a_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Slide A created")


def create_overfitting_reduction():
    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, single_tree_gaps, width, label='Single Tree', 
                   color=COLORS['single_tree'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, our_rf_gaps, width, label='Random Forest', 
                   color=COLORS['our_rf'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Overfitting Gap (Train R² - Test R²)', fontweight='bold')
    ax.set_title('Ensemble Reduces Overfitting', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.axhline(y=0, color='black', linestyle='--')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.02 if height >= 0 else -0.05),
                   f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide_b_overfitting_reduction.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'slide_b_overfitting_reduction.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Slide B created")


def create_n_estimators_impact():
    fig, ax = plt.subplots(figsize=(11, 7))
    
    ax.plot(n_estimators_values, n_estimators_train_r2, 'o--', label='Train R²', 
            color=COLORS['train'], linewidth=2.5, markersize=9, alpha=0.7)
    ax.plot(n_estimators_values, n_estimators_test_r2, 's-', label='Test R²', 
            color=COLORS['test'], linewidth=2.5, markersize=9)
    
    optimal_idx = n_estimators_test_r2.index(max(n_estimators_test_r2))
    optimal_n = n_estimators_values[optimal_idx]
    ax.axvline(x=optimal_n, color=COLORS['our_rf'], linestyle=':', linewidth=2.5, label=f'Optimal: {optimal_n}')
    ax.plot(optimal_n, n_estimators_test_r2[optimal_idx], 'g*', markersize=20)
    
    ax.set_xlabel('Number of Trees (n_estimators)', fontweight='bold')
    ax.set_ylabel('R² Score', fontweight='bold')
    ax.set_title('Impact of Ensemble Size (Ames Housing)', fontweight='bold', pad=20)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide_c_n_estimators_impact.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'slide_c_n_estimators_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Slide C created")


def create_max_depth_impact():
    fig, ax1 = plt.subplots(figsize=(11, 7))
    
    ax1.plot(max_depth_values, max_depth_train_r2, 'o--', label='Train R²', 
            color=COLORS['train'], linewidth=2.5, markersize=9)
    ax1.plot(max_depth_values, max_depth_test_r2, 's-', label='Test R²', 
            color=COLORS['test'], linewidth=2.5, markersize=9)
    ax1.set_xlabel('Maximum Tree Depth', fontweight='bold')
    ax1.set_ylabel('R² Score', fontweight='bold')
    ax1.grid(alpha=0.3)
    
    ax2 = ax1.twinx()
    gaps = [t - te for t, te in zip(max_depth_train_r2, max_depth_test_r2)]
    ax2.plot(max_depth_values, gaps, '^-', label='Overfitting Gap', color='#e74c3c', linewidth=2.5, markersize=8)
    ax2.set_ylabel('Overfitting Gap', color='#e74c3c', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    ax1.set_title('Bias-Variance Trade-off: max_depth (Ames Housing)', fontweight='bold', pad=20)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide_d_max_depth_impact.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'slide_d_max_depth_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Slide D created")


def create_ccp_tradeoff():
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    color1 = COLORS['test']
    ax1.set_xlabel('CCP Alpha (log scale)', fontweight='bold')
    ax1.set_ylabel('Test R²', color=color1, fontweight='bold')
    
    # Handle alpha=0 for log scale
    alphas_plot = [max(a, 1e5) for a in ccp_alphas]
    line1, = ax1.semilogx(alphas_plot, ccp_test_r2, 'o-', color=color1, linewidth=3, markersize=10, label='Test R²')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = '#9b59b6'
    ax2.set_ylabel('Number of Leaves', color=color2, fontweight='bold')
    line2, = ax2.semilogx(alphas_plot, ccp_n_leaves, 's-', color=color2, linewidth=3, markersize=10, label='Tree Size')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    optimal_idx = ccp_test_r2.index(max(ccp_test_r2))
    ax1.plot(alphas_plot[optimal_idx], ccp_test_r2[optimal_idx], 'g*', markersize=25, zorder=10, label='Optimal α')
    
    ax1.set_title('Cost-Complexity Pruning Trade-off (Ames Housing)', fontweight='bold', pad=20)
    lines = [line1, line2]
    labels = ['Test R²', 'Tree Size']
    ax1.legend(lines, labels, loc='center right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide_f_ccp_tradeoff.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'slide_f_ccp_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Slide F created")


if __name__ == "__main__":
    print("=" * 50)
    print("Creating Presentation Visualizations")
    print("=" * 50)
    print()
    create_performance_comparison()
    create_overfitting_reduction()
    create_n_estimators_impact()
    create_max_depth_impact()
    create_ccp_tradeoff()
    print()
    print(f"✓ All plots saved to: {OUTPUT_DIR.resolve()}")
    print("  - PDF format (for presentations)")
    print("  - PNG format (for backup)")


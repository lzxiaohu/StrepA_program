# plot_prior_distribution.py
# Visualize the distribution of R0 and sigma in the 500k simulation bank

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_prior_distribution(
    R0_file='../../experimental_data/from_260430/R0.csv',
    sigma_file='../../experimental_data/from_260430/sigma.csv',
    true_R0=None,
    true_sigma=None,
    save_path='../../figures/from_260430/'
):
    """
    Plot the distribution of R0 and sigma from the simulation bank.
    
    Creates 4 plots:
    1. 2D scatter plot (R0 vs sigma)
    2. 2D density plot (heatmap)
    3. Marginal distributions (histograms)
    4. Combined: scatter + marginals
    """
    
    # Load data
    print("Loading data...")
    R0 = pd.read_csv(R0_file, header=None).values.ravel()
    sigma = pd.read_csv(sigma_file, header=None).values.ravel()
    
    print(f"✓ Loaded {len(R0):,} samples")
    print(f"  R0 range: [{R0.min():.3f}, {R0.max():.3f}]")
    print(f"  Sigma range: [{sigma.min():.3f}, {sigma.max():.3f}]")
    
    # Create output directory
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # ========================================================================
    # PLOT 1: Simple 2D Scatter
    # ========================================================================
    print("\nGenerating Plot 1: 2D Scatter...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sample for visualization (plot subset if too many points)
    n_plot = min(50000, len(R0))
    indices = np.random.choice(len(R0), n_plot, replace=False)
    
    ax.scatter(R0[indices], sigma[indices], alpha=0.3, s=10, c='blue', label='Simulations')
    
    # Mark true values if provided
    if true_R0 is not None and true_sigma is not None:
        ax.scatter(true_R0, true_sigma, s=300, c='red', marker='*',
                  edgecolors='black', linewidths=2, label='True value', zorder=10)
    
    ax.set_xlabel('R0', fontsize=14)
    ax.set_ylabel('sigma', fontsize=14)
    ax.set_title(f'Prior Distribution: R0 vs sigma\n({len(R0):,} samples, showing {n_plot:,})',
                fontsize=16, fontweight='bold')
    ax.set_xlim(1, 8)
    ax.set_ylim(0.2, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}prior_scatter.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}prior_scatter.png")
    plt.close()
    
    # ========================================================================
    # PLOT 2: 2D Density Heatmap
    # ========================================================================
    print("\nGenerating Plot 2: 2D Density Heatmap...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create 2D histogram
    h = ax.hist2d(R0, sigma, bins=100, cmap='Blues', cmin=1)
    plt.colorbar(h[3], ax=ax, label='Number of samples')
    
    # Mark true values
    if true_R0 is not None and true_sigma is not None:
        ax.scatter(true_R0, true_sigma, s=300, c='red', marker='*',
                  edgecolors='black', linewidths=2, label='True value', zorder=10)
        ax.legend(fontsize=12)
    
    ax.set_xlabel('R0', fontsize=14)
    ax.set_ylabel('sigma', fontsize=14)
    ax.set_title(f'Prior Density: R0 vs sigma\n({len(R0):,} samples)',
                fontsize=16, fontweight='bold')
    ax.set_xlim(1, 8)
    ax.set_ylim(0.2, 1.0)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}prior_density.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}prior_density.png")
    plt.close()
    
    # ========================================================================
    # PLOT 3: Marginal Distributions
    # ========================================================================
    print("\nGenerating Plot 3: Marginal Distributions...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # R0 histogram
    ax1.hist(R0, bins=100, color='blue', alpha=0.7, edgecolor='black')
    if true_R0 is not None:
        ax1.axvline(true_R0, color='red', linestyle='--', linewidth=2, label=f'True R0 = {true_R0}')
        ax1.legend(fontsize=12)
    ax1.set_xlabel('R0', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax1.set_title(f'R0 Distribution\n(mean={R0.mean():.2f}, std={R0.std():.2f})',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Sigma histogram
    ax2.hist(sigma, bins=100, color='green', alpha=0.7, edgecolor='black')
    if true_sigma is not None:
        ax2.axvline(true_sigma, color='red', linestyle='--', linewidth=2, label=f'True σ = {true_sigma}')
        ax2.legend(fontsize=12)
    ax2.set_xlabel('sigma', fontsize=14)
    ax2.set_ylabel('Frequency', fontsize=14)
    ax2.set_title(f'Sigma Distribution\n(mean={sigma.mean():.2f}, std={sigma.std():.2f})',
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}prior_marginals.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}prior_marginals.png")
    plt.close()
    
    # ========================================================================
    # PLOT 4: Joint Plot with Marginals (seaborn style)
    # ========================================================================
    print("\nGenerating Plot 4: Joint Plot with Marginals...")
    
    # Sample for faster plotting
    n_sample = min(10000, len(R0))
    sample_indices = np.random.choice(len(R0), n_sample, replace=False)
    
    # Create DataFrame for seaborn
    df = pd.DataFrame({
        'R0': R0[sample_indices],
        'sigma': sigma[sample_indices]
    })
    
    # Joint plot
    g = sns.jointplot(data=df, x='R0', y='sigma', kind='scatter',
                     alpha=0.3, s=10, height=10)
    
    # Add true values
    if true_R0 is not None and true_sigma is not None:
        g.ax_joint.scatter(true_R0, true_sigma, s=300, c='red', marker='*',
                          edgecolors='black', linewidths=2, zorder=10, label='True value')
        g.ax_joint.legend(fontsize=12)
    
    g.ax_joint.set_xlim(1, 8)
    g.ax_joint.set_ylim(0.2, 1.0)
    g.fig.suptitle(f'Prior Distribution with Marginals\n({len(R0):,} samples)',
                   fontsize=16, fontweight='bold', y=1.02)
    
    plt.savefig(f'{save_path}prior_joint.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}prior_joint.png")
    plt.close()
    
    # ========================================================================
    # Statistics Summary
    # ========================================================================
    print("\n" + "="*70)
    print("DISTRIBUTION STATISTICS")
    print("="*70)
    print(f"\nR0:")
    print(f"  Range:  [{R0.min():.3f}, {R0.max():.3f}]")
    print(f"  Mean:   {R0.mean():.3f}")
    print(f"  Median: {np.median(R0):.3f}")
    print(f"  Std:    {R0.std():.3f}")
    
    print(f"\nSigma:")
    print(f"  Range:  [{sigma.min():.3f}, {sigma.max():.3f}]")
    print(f"  Mean:   {sigma.mean():.3f}")
    print(f"  Median: {np.median(sigma):.3f}")
    print(f"  Std:    {sigma.std():.3f}")
    
    if true_R0 is not None and true_sigma is not None:
        print(f"\nTrue values:")
        print(f"  R0:    {true_R0}")
        print(f"  Sigma: {true_sigma}")
        
        # Count samples near true values
        tolerance_R0 = 0.2
        tolerance_sigma = 0.1
        near_true = ((R0 >= true_R0 - tolerance_R0) & (R0 <= true_R0 + tolerance_R0) &
                    (sigma >= true_sigma - tolerance_sigma) & (sigma <= true_sigma + tolerance_sigma))
        n_near = near_true.sum()
        
        print(f"\nSamples near true values:")
        print(f"  Within ±{tolerance_R0} of R0 and ±{tolerance_sigma} of sigma: {n_near:,} ({100*n_near/len(R0):.2f}%)")
    
    print("="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    plot_prior_distribution(
        R0_file='../../experimental_data/from_260430/R0.csv',
        sigma_file='../../experimental_data/from_260430/sigma.csv',
        true_R0=None,
        true_sigma=None,
        save_path='../../figures/from_260430/prior/'
    )
    
    print("\n✅ All plots generated successfully!")
    print("\nGenerated plots:")
    print("  1. prior_scatter.png   - 2D scatter plot")
    print("  2. prior_density.png   - 2D density heatmap")
    print("  3. prior_marginals.png - Marginal histograms")
    print("  4. prior_joint.png     - Joint plot with marginals")
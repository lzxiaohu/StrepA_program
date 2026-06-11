# plot_contour_2d.py
# 2D contour plot of R0 vs sigma for ABC posterior

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


def plot_contour_2d(
    file_dists='../../experimental_data/from_260528/dists_observations_recal.csv',
    file_R0='../../experimental_data/from_260528/R0.csv',
    file_sigma='../../experimental_data/from_260528/sigma.csv',
    percentile=1.0,
    true_R0=None,
    true_sigma=None,
    title="ABC Posterior - 2D Contour",
    save_path='../../figures/from_260528/ppc/observations/',
    save_filename='contour_2d.png',
    xlim=(1, 8),
    ylim=(0.2, 1.0),
    bw_method=0.3,       # KDE bandwidth
    n_levels=10,         # Number of contour levels
    show_scatter=True,   # Show scatter points under contour
):
    """
    Plot 2D contour of R0 vs sigma for selected percentile.
    
    Parameters:
    -----------
    file_dists : str
        CSV file with distances
    file_R0 : str
        CSV file with R0 values
    file_sigma : str
        CSV file with sigma values
    percentile : float
        Percentile threshold (e.g., 1.0 means closest 1%)
    true_R0 : float, optional
        True R0 value to mark
    true_sigma : float, optional
        True sigma value to mark
    title : str
        Figure title
    save_path : str
        Directory to save figure
    save_filename : str
        Filename for saved figure
    xlim : tuple
        X-axis limits (R0 range)
    ylim : tuple
        Y-axis limits (sigma range)
    bw_method : float
        KDE bandwidth (smaller = sharper contours)
    n_levels : int
        Number of contour levels
    show_scatter : bool
        Whether to show scatter points underneath contours
    """
    
    print("="*70)
    print(f"PLOTTING 2D CONTOUR: R0 vs SIGMA")
    print("="*70)
    
    # Load data
    distances = pd.read_csv(file_dists, header=None).values.ravel()
    R0_array = pd.read_csv(file_R0, header=None).values.ravel()
    sigma_array = pd.read_csv(file_sigma, header=None).values.ravel()
    
    print(f"Loaded {len(distances):,} samples")
    
    # Select samples at percentile
    threshold = np.percentile(distances, percentile)
    selected = distances <= threshold
    
    selected_R0 = R0_array[selected]
    selected_sigma = sigma_array[selected]
    n_selected = selected.sum()
    
    print(f"Percentile {percentile}%: {n_selected:,} samples selected")
    print(f"R0 range:    [{selected_R0.min():.3f}, {selected_R0.max():.3f}]")
    print(f"Sigma range: [{selected_sigma.min():.3f}, {selected_sigma.max():.3f}]")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot underneath (optional)
    if show_scatter:
        ax.scatter(selected_R0, selected_sigma,
                  alpha=0.2, s=10, c='gray', zorder=1,
                  label='Samples')
    
    # 2D KDE
    print(f"\nCalculating 2D KDE (bw_method={bw_method})...")
    xy = np.vstack([selected_R0, selected_sigma])
    kde = stats.gaussian_kde(xy, bw_method=bw_method)
    
    # Evaluate KDE on a grid
    x_grid = np.linspace(xlim[0], xlim[1], 200)
    y_grid = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    
    # Find peak density (mode)
    peak_idx = np.unravel_index(np.argmax(Z), Z.shape)
    peak_R0 = X[peak_idx]
    peak_sigma = Y[peak_idx]
    peak_density = Z[peak_idx]
    
    print(f"\nPeak density location:")
    print(f"  R0    = {peak_R0:.4f}")
    print(f"  sigma = {peak_sigma:.4f}")
    print(f"  density = {peak_density:.6f}")
    
    print(f"KDE grid: 200x200")
    
    # Filled contour
    contourf = ax.contourf(X, Y, Z, levels=n_levels, 
                          cmap='YlOrRd', alpha=0.8, zorder=2)
    
    # Contour lines
    contour = ax.contour(X, Y, Z, levels=n_levels,
                        colors='black', linewidths=0.5, alpha=0.5, zorder=3)
    
    # Colorbar
    cbar = plt.colorbar(contourf, ax=ax, pad=0.02)
    cbar.set_label('Density', fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    # Mark peak density
    ax.scatter(peak_R0, peak_sigma,
              s=200, c='blue', marker='+',
              linewidths=3, zorder=10,
              label=f'Peak (R0={peak_R0:.3f}, σ={peak_sigma:.3f})')
    
    ax.annotate(f'R0={peak_R0:.3f}\nσ={peak_sigma:.3f}',
               xy=(peak_R0, peak_sigma),
               xytext=(15, 15), textcoords='offset points',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue', alpha=0.9),
               arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    # Mark true values if provided
    if true_R0 is not None and true_sigma is not None:
        ax.scatter(true_R0, true_sigma,
                  s=300, c='blue', marker='*',
                  edgecolors='black', linewidths=1.5,
                  zorder=10, label=f'True (R0={true_R0}, σ={true_sigma})')
    
    # Labels
    ax.set_xlabel('R0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sigma', fontsize=14, fontweight='bold')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f'{title}\n{percentile}% percentile ({n_selected:,} samples)',
                fontsize=15, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_file = f"{save_path}{save_filename}"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {save_file}")
    plt.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    plot_contour_2d(
        file_dists='../../experimental_data/from_260528/dists_observations_recal.csv',
        file_R0='../../experimental_data/from_260528/R0.csv',
        file_sigma='../../experimental_data/from_260528/sigma.csv',
        percentile=1.0,
        true_R0=None,       # Set to your true value if known
        true_sigma=None,    # Set to your true value if known
        title="ABC Posterior",
        save_path='../../figures/from_260528/ppc/observations/',
        save_filename='contour_2d_p1.png',
        xlim=(1, 4),
        ylim=(0.2, 1.0),
        bw_method=0.3,
        n_levels=10,
        show_scatter=True
    )
    
    print("\n✅ Done!")
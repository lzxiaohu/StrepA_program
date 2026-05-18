import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_dots_percentile_colors(file_dists='dists.csv', 
                                file_R0='R0.csv', 
                                file_sigma='sigma.csv',
                                percentile=1,
                                true_R0=None,
                                true_sigma=None,
                                title="R0 vs sigma - Color by Distance",
                                save_path='../../experimental_data/from_260312/',
                                xlim=(1, 8),
                                ylim=(0.2, 1.0),
                                cmap='viridis_r'):  # 'viridis_r' = purple (close) to yellow (far)
    """
    Plot R0 vs sigma with colors representing distance from standard point.
    
    Parameters:
    -----------
    file_dists : str
        CSV file with distances (single column)
    file_R0 : str
        CSV file with R0 samples
    file_sigma : str
        CSV file with sigma samples
    percentile : float
        Percentile threshold (e.g., 1 means keep closest 1%)
    true_R0 : float, optional
        True R0 value to mark on plot
    true_sigma : float, optional
        True sigma value to mark on plot
    title : str
        Figure title
    save_path : str
        Directory to save figure
    xlim : tuple
        X-axis limits (R0 range)
    ylim : tuple
        Y-axis limits (sigma range)
    cmap : str
        Colormap name. Options:
        - 'viridis_r': purple (close) → yellow (far)
        - 'RdYlGn_r': red (far) → green (close)
        - 'coolwarm': blue (close) → red (far)
        - 'plasma_r': purple (close) → yellow (far)
    """
    
    # Load data
    print("Loading data...")
    distances = pd.read_csv(file_dists, header=None).values.ravel()
    data_R0 = pd.read_csv(file_R0, header=None).values.ravel()
    data_sigma = pd.read_csv(file_sigma, header=None).values.ravel()
    
    print(f"✓ Loaded {len(distances):,} samples")
    print(f"  Distance range: [{distances.min():.6f}, {distances.max():.6f}]")
    
    # Select samples based on percentile
    threshold = np.percentile(distances, percentile)
    selected_indices = np.where(distances <= threshold)[0]
    n_selected = len(selected_indices)
    
    print(f"\nPercentile {percentile}%:")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Selected: {n_selected:,} samples ({100*n_selected/len(distances):.2f}%)")
    
    # Get selected data
    selected_R0 = data_R0[selected_indices]
    selected_sigma = data_sigma[selected_indices]
    selected_dist = distances[selected_indices]
    
    print(f"  Distance range of selected: [{selected_dist.min():.6f}, {selected_dist.max():.6f}]")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create scatter plot with color mapping
    scatter = ax.scatter(
        selected_R0, 
        selected_sigma, 
        c=selected_dist,  # Color by distance
        cmap=cmap,
        s=50,  # Point size
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5,
        vmin=selected_dist.min(),
        vmax=selected_dist.max()
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Distance from Standard Point', pad=0.02)
    cbar.ax.tick_params(labelsize=11)
    
    # Mark true values if provided
    if true_R0 is not None and true_sigma is not None:
        ax.scatter(true_R0, true_sigma, s=400, c='red', marker='*',
                  edgecolors='black', linewidths=3, label='True value', zorder=10)
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
    
    # Set fixed ranges
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Labels and title
    ax.set_xlabel('R0', fontsize=14, fontweight='bold')
    ax.set_ylabel('sigma', fontsize=14, fontweight='bold')
    ax.set_title(
        f'{title}\n'
        f'{percentile}% closest to standard point '
        f'({n_selected:,} samples, dist ≤ {threshold:.4f})',
        fontsize=16, fontweight='bold', pad=20
    )
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text box
    textstr = f'Distance Range:\n  Min: {selected_dist.min():.6f}\n  Max: {selected_dist.max():.6f}\n  Mean: {selected_dist.mean():.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save figure
    import os
    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_path}{title.replace(' ', '_').replace('-', '')}_p{percentile}.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {save_file}")
    plt.close()
 

def plot_dots_multiple_percentiles(file_dists='dists.csv', 
                                   file_R0='R0.csv', 
                                   file_sigma='sigma.csv',
                                   percentiles=[100, 80, 60, 40, 20, 10],
                                   true_R0=None,
                                   true_sigma=None,
                                   title="R0 vs sigma - Multiple Percentiles",
                                   save_path='../../experimental_data/from_260312/',
                                   xlim=(1, 8),        # Fixed x-axis range
                                   ylim=(0.2, 1.0)):   # Fixed y-axis range
    """
    Create 6 subplots showing R0 vs sigma for different distance percentiles.
    
    Parameters:
    -----------
    file_dists : str
        CSV file with distances (single column)
    file_R0 : str
        CSV file with R0 samples
    file_sigma : str
        CSV file with sigma samples
    percentiles : list
        List of percentiles to plot [100, 80, 60, 40, 20, 10]
    true_R0 : float, optional
        True R0 value to mark on plots
    true_sigma : float, optional
        True sigma value to mark on plots
    title : str
        Overall figure title
    save_path : str
        Directory to save figure
    xlim : tuple
        X-axis limits (R0 range), default (1, 8)
    ylim : tuple
        Y-axis limits (sigma range), default (0.2, 1.0)
    """
    # Load data
    distances = pd.read_csv(file_dists, header=None).values.ravel()
    data_R0 = pd.read_csv(file_R0, header=None).values.ravel()
    data_sigma = pd.read_csv(file_sigma, header=None).values.ravel()
    
    print(f"Loaded {len(distances)} samples")
    
    # Create figure with 6 subplots (2 rows x 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()  # Flatten to 1D array for easy indexing
    
    # Plot for each percentile
    for idx, percentile in enumerate(percentiles):
        ax = axes[idx]
        
        # Calculate threshold
        if percentile == 100:
            # Keep all samples
            selected_indices = np.arange(len(distances))
        else:
            threshold = np.percentile(distances, percentile)
            selected_indices = np.where(distances <= threshold)[0]
        
        n_selected = len(selected_indices)
        print(f"Percentile {percentile}: {n_selected} samples selected")
        
        # Select corresponding samples
        selected_R0 = data_R0[selected_indices]
        selected_sigma = data_sigma[selected_indices]
        
        # Scatter plot
        ax.scatter(selected_R0, selected_sigma, alpha=0.5, s=20, c='blue')
        
        # Mark true values if provided
        if true_R0 is not None and true_sigma is not None:
            ax.scatter(true_R0, true_sigma, s=200, c='red', marker='*',
                      edgecolors='black', linewidths=2, label='True value', zorder=5)
            ax.legend(loc='best')
        
        # Set fixed axis ranges
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Labels and title
        ax.set_xlabel('R0', fontsize=11)
        ax.set_ylabel('sigma', fontsize=11)
        ax.set_title(f'{percentile}th percentile\n({n_selected} samples, {n_selected/len(distances)*100:.1f}%)',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    import os
    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_path}{title.replace(' ', '_').replace('-', '')}.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_file}")
    plt.close()


def analyze_distances(
    filepath: str,
    standard_point: tuple,
    weights: tuple = None,
    metric: str = "euclidean",
    p: float = 3,
    output_path: str = "output.csv"
) -> pd.DataFrame:
    """
    Load a headerless CSV with 4 features, apply Min-Max normalization,
    and compute weighted distance of each row to a standard point.

    Parameters
    ----------
    filepath       : path to the input CSV file (no header, 4 columns)
    standard_point : tuple of (A0, B0, C0, D0)
    weights        : optional tuple of (wA, wB, wC, wD)
    metric         : distance metric (euclidean, manhattan, chebyshev, minkowski, cosine)
    p              : power for minkowski distance
    output_path    : path to save the result CSV

    Returns
    -------
    pd.DataFrame with normalized columns A, B, C, D and a 'distance' column
    """
    # 1. Load
    df = pd.read_csv(filepath, header=None, names=["A", "B", "C", "D"])

    # 2. Min-Max normalization
    col_min = df.min()
    col_max = df.max()
    df_norm = (df - col_min) / (col_max - col_min)

    # 3. Normalize the standard point on the same scale
    standard = np.array(standard_point)
    standard_norm = (standard - col_min.values) / (col_max.values - col_min.values)

    # 4. Resolve weights (normalize so they sum to 1)
    if weights is not None:
        w = np.array(weights, dtype=float)
        if len(w) != 4:
            raise ValueError("weights must have exactly 4 values (wA, wB, wC, wD).")
        if np.any(w < 0):
            raise ValueError("All weights must be non-negative.")
        w = w / w.sum()
    else:
        w = np.array([0.25, 0.25, 0.25, 0.25])

    # 5. Weighted distance
    diff = (df_norm[["A", "B", "C", "D"]] - standard_norm).values
    
    if metric == "euclidean":
        dist = np.sqrt((w * diff ** 2).sum(axis=1))
    elif metric == "manhattan":
        dist = (w * np.abs(diff)).sum(axis=1)
    elif metric == "chebyshev":
        dist = (w * np.abs(diff)).max(axis=1)
    elif metric == "minkowski":
        dist = ((w * np.abs(diff) ** p).sum(axis=1)) ** (1 / p)
    elif metric == "cosine":
        dot = (w * df_norm[["A", "B", "C", "D"]].values * standard_norm).sum(axis=1)
        norm_x = np.sqrt((w * df_norm[["A", "B", "C", "D"]].values ** 2).sum(axis=1))
        norm_x0 = np.sqrt((w * standard_norm ** 2).sum())
        dist = 1 - dot / (norm_x * norm_x0)
    else:
        raise ValueError(f"Unknown metric '{metric}'")

    df_norm["distance"] = dist

    # 6. Save & return
    df_norm[["distance"]].to_csv(output_path, index=False, header=False)
    print(f"Done. Results saved to '{output_path}'.")
    return df_norm


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # 1. Calculate distances with Euclidean metric
    result = analyze_distances(
        filepath="../../experimental_data/from_260430/summary_stats_normalized.csv",
        standard_point=(0.02690748, 0.03816928, 0.0603244, 0.46641378),  # Fixed: added commas
        weights=(0.9, 0.9, 0.1, 0.1),  # Fixed: added commas
        metric="euclidean",
        p=3,
        output_path="../../experimental_data/from_260430/dists_observations_recal.csv"
    )
    
    # 2. Plot with higher percentiles (10%, 5%, 4%, 3%, 2%, 1%)
    plot_dots_multiple_percentiles(
        file_dists='../../experimental_data/from_260430/dists_observations_recal.csv',
        file_R0='../../experimental_data/from_260430/R0.csv',
        file_sigma='../../experimental_data/from_260430/sigma.csv',
        percentiles=[10, 5, 4, 3, 2, 1],
        true_R0=0.0,
        true_sigma=0.0,
        title="R0 vs sigma - Multiple Percentiles-euclidean1",
        save_path="../../figures/from_260430/ppc/observations/",
        xlim=(1, 8),      # R0 range
        ylim=(0.2, 1.0)   # sigma range
    )
    
    # 3. Plot with lower percentiles (2%, 1%, 0.5%, 0.4%, 0.2%, 0.1%)
    plot_dots_multiple_percentiles(
        file_dists='../../experimental_data/from_260430/dists_observations_recal.csv',
        file_R0='../../experimental_data/from_260430/R0.csv',
        file_sigma='../../experimental_data/from_260430/sigma.csv',
        percentiles=[2, 1, 0.5, 0.4, 0.2, 0.1],
        true_R0=0.0,
        true_sigma=0.0,
        title="R0 vs sigma - Multiple Percentiles-euclidean",
        save_path="../../figures/from_260430/ppc/observations/",
        xlim=(1, 8),      # R0 range
        ylim=(0.2, 1.0)   # sigma range
    )
    
plot_dots_percentile_colors(
        file_dists='../../experimental_data/from_260430/dists_observations_recal.csv',
        file_R0='../../experimental_data/from_260430/R0.csv',
        file_sigma='../../experimental_data/from_260430/sigma.csv',
        percentile=1,  # Show closest 5%
        true_R0=0.0,
        true_sigma=0.0,
        title="ABC Posterior - Top 1%",
        save_path="../../figures/from_260430/ppc/observations/",
        xlim=(1, 8),
        ylim=(0.2, 1.0),
        cmap='viridis_r'  # Purple (close) to yellow (far)
    )

plot_dots_percentile_colors(
        file_dists='../../experimental_data/from_260430/dists_observations_recal.csv',
        file_R0='../../experimental_data/from_260430/R0.csv',
        file_sigma='../../experimental_data/from_260430/sigma.csv',
        percentile=0.1,  # Show closest 5%
        true_R0=0.0,
        true_sigma=0.0,
        title="ABC Posterior - Top 0.1%",
        save_path="../../figures/from_260430/ppc/observations/",
        xlim=(1, 8),
        ylim=(0.2, 1.0),
        cmap='viridis_r'  # Purple (close) to yellow (far)
    )

plot_dots_percentile_colors(
        file_dists='../../experimental_data/from_260430/dists_observations_recal.csv',
        file_R0='../../experimental_data/from_260430/R0.csv',
        file_sigma='../../experimental_data/from_260430/sigma.csv',
        percentile=0.05,  # Show closest 5%
        true_R0=0.0,
        true_sigma=0.0,
        title="ABC Posterior - Top 0.05%",
        save_path="../../figures/from_260430/ppc/observations/",
        xlim=(1, 8),
        ylim=(0.2, 1.0),
        cmap='viridis_r'  # Purple (close) to yellow (far)
    )

plot_dots_percentile_colors(
        file_dists='../../experimental_data/from_260430/dists_observations_recal.csv',
        file_R0='../../experimental_data/from_260430/R0.csv',
        file_sigma='../../experimental_data/from_260430/sigma.csv',
        percentile=0.01,  # Show closest 5%
        true_R0=0.0,
        true_sigma=0.0,
        title="ABC Posterior - Top 0.01%",
        save_path="../../figures/from_260430/ppc/observations/",
        xlim=(1, 8),
        ylim=(0.2, 1.0),
        cmap='viridis_r'  # Purple (close) to yellow (far)
    )

print("\n✅ All plots generated successfully!")
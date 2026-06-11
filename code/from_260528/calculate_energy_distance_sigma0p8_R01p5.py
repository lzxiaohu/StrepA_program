# calculate_energy_distance_sigma0p8_R01p5.py
# Calculate energy distance between simulation posterior and observation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


def energy_distance(X, Y):
    """
    Calculate energy distance between two sets of points.
    
    E(X, Y) = 2 * E[|X - Y|] - E[|X - X'|] - E[|Y - Y'|]
    """
    
    cross_dist = np.mean(
        np.sqrt(np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1))
    )
    xx_dist = np.mean(
        np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1))
    )
    yy_dist = np.mean(
        np.sqrt(np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=-1))
    )
    
    return 2 * cross_dist - xx_dist - yy_dist


def calculate_energy_distance_abc(
    file_dists='../../experimental_data/from_260528/dists_sigma0p8_R01p5_recal.csv',
    file_R0='../../experimental_data/from_260528/R0.csv',
    file_sigma='../../experimental_data/from_260528/sigma.csv',
    percentile=1.0,
    n_subsample=1000,
    random_seed=42
):
    """
    Calculate energy distance between posterior (accepted samples) and prior (all samples).
    """
    
    print("="*70)
    print("CALCULATING ENERGY DISTANCE: POSTERIOR vs PRIOR")
    print("="*70)
    
    np.random.seed(random_seed)
    
    # Load data
    distances = pd.read_csv(file_dists, header=None).values.ravel()
    R0_array = pd.read_csv(file_R0, header=None).values.ravel()
    sigma_array = pd.read_csv(file_sigma, header=None).values.ravel()
    
    n_total = len(distances)
    print(f"Total samples (prior): {n_total:,}")
    
    # Posterior
    threshold = np.percentile(distances, percentile)
    selected = distances <= threshold
    post_R0 = R0_array[selected]
    post_sigma = sigma_array[selected]
    n_posterior = selected.sum()
    
    print(f"Posterior ({percentile}%): {n_posterior:,} samples")
    print(f"  R0:    mean={post_R0.mean():.3f}, std={post_R0.std():.3f}")
    print(f"  sigma: mean={post_sigma.mean():.3f}, std={post_sigma.std():.3f}")
    
    # Prior
    print(f"\nPrior: {n_total:,} samples")
    print(f"  R0:    mean={R0_array.mean():.3f}, std={R0_array.std():.3f}")
    print(f"  sigma: mean={sigma_array.mean():.3f}, std={sigma_array.std():.3f}")
    
    # Subsample
    if n_posterior > n_subsample:
        idx = np.random.choice(n_posterior, n_subsample, replace=False)
        post_R0 = post_R0[idx]
        post_sigma = post_sigma[idx]
        print(f"\nPosterior subsampled: {n_posterior:,} → {n_subsample}")
    else:
        print(f"\nPosterior using ALL {n_posterior} samples (< {n_subsample})")
    
    if n_total > n_subsample:
        idx_prior = np.random.choice(n_total, n_subsample, replace=False)
        prior_R0 = R0_array[idx_prior]
        prior_sigma = sigma_array[idx_prior]
        print(f"Prior subsampled: {n_total:,} → {n_subsample}")
    else:
        prior_R0 = R0_array
        prior_sigma = sigma_array
        print(f"Prior using ALL {n_total} samples")
    
    X = np.column_stack([post_R0, post_sigma])
    Y = np.column_stack([prior_R0, prior_sigma])
    
    print(f"Posterior shape: {X.shape}")
    print(f"Prior shape:     {Y.shape}")
    
    print(f"\nCalculating energy distance...")
    ed = energy_distance(X, Y)
    
    print(f"\n{'='*70}")
    print(f"ENERGY DISTANCE RESULT")
    print(f"{'='*70}")
    print(f"Energy distance (posterior vs prior): {ed:.6f}")
    print(f"\nInterpretation:")
    print(f"  Large value → posterior very different from prior")
    print(f"               → ABC learned a lot from the observation")
    print(f"  Small value → posterior similar to prior")
    print(f"               → ABC learned little from the observation")
    
    return ed


def plot_energy_distance(
    file_dists='../../experimental_data/from_260528/dists_sigma0p8_R01p5_recal.csv',
    file_R0='../../experimental_data/from_260528/R0.csv',
    file_sigma='../../experimental_data/from_260528/sigma.csv',
    percentile=1.0,
    true_R0=None,        # ← True R0 value to mark
    true_sigma=None,     # ← True sigma value to mark
    xlim=(1, 4),
    ylim=(0.2, 1.0),
    bw_method='silverman',
    title="ABC Posterior",
    save_path='../../figures/from_260528/ppc/sigma0p8/R01p5/',
    save_filename='energy_distance_plot.png'
):
    """
    Plot scatter and contour side by side for the posterior samples.
    Marks true R0 and sigma values if provided.
    """
    
    print("="*70)
    print("PLOTTING SCATTER + CONTOUR")
    print("="*70)
    
    # Load data
    distances = pd.read_csv(file_dists, header=None).values.ravel()
    R0_array = pd.read_csv(file_R0, header=None).values.ravel()
    sigma_array = pd.read_csv(file_sigma, header=None).values.ravel()
    
    # Select samples at percentile
    threshold = np.percentile(distances, percentile)
    selected = distances <= threshold
    
    selected_R0 = R0_array[selected]
    selected_sigma = sigma_array[selected]
    n_selected = selected.sum()
    
    print(f"Percentile {percentile}%: {n_selected:,} samples")
    
    if true_R0 is not None and true_sigma is not None:
        print(f"True values: R0={true_R0}, sigma={true_sigma}")
    
    # Find peak from KDE
    xy = np.vstack([selected_R0, selected_sigma])
    kde = stats.gaussian_kde(xy, bw_method=bw_method)
    
    x_grid = np.linspace(xlim[0], xlim[1], 200)
    y_grid = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    
    peak_idx = np.unravel_index(np.argmax(Z), Z.shape)
    peak_R0 = X[peak_idx]
    peak_sigma = Y[peak_idx]
    
    print(f"Peak: R0={peak_R0:.4f}, sigma={peak_sigma:.4f}")
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # ========================================================================
    # Left: Scatter plot
    # ========================================================================
    sc = ax1.scatter(selected_R0, selected_sigma,
                    c=distances[selected],
                    cmap='YlOrRd_r',
                    alpha=0.4, s=10, zorder=2)
    
    cbar1 = plt.colorbar(sc, ax=ax1, pad=0.02)
    cbar1.set_label('Distance', fontsize=12, fontweight='bold')
    
    # Mark peak
    ax1.scatter(peak_R0, peak_sigma,
               s=300, c='blue', marker='+',
               linewidths=3, zorder=10,
               label=f'Peak (R0={peak_R0:.3f}, σ={peak_sigma:.3f})')
    ax1.annotate(f'R0={peak_R0:.3f}\nσ={peak_sigma:.3f}',
                xy=(peak_R0, peak_sigma),
                xytext=(15, 15), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    # Mark true values
    if true_R0 is not None and true_sigma is not None:
        ax1.scatter(true_R0, true_sigma,
                   s=300, c='red', marker='*',
                   edgecolors='black', linewidths=1.5,
                   zorder=11, label=f'True (R0={true_R0}, σ={true_sigma})')
        ax1.annotate(f'True\nR0={true_R0}\nσ={true_sigma}',
                    xy=(true_R0, true_sigma),
                    xytext=(-60, 15), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    ax1.set_xlabel('R0', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Sigma', fontsize=13, fontweight='bold')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_title(f'Scatter\n({n_selected:,} samples)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, framealpha=0.9)
    
    # ========================================================================
    # Right: Contour plot
    # ========================================================================
    contourf = ax2.contourf(X, Y, Z, levels=10,
                           cmap='YlOrRd', alpha=0.8, zorder=2)
    ax2.contour(X, Y, Z, levels=10,
               colors='black', linewidths=0.5, alpha=0.5, zorder=3)
    
    cbar2 = plt.colorbar(contourf, ax=ax2, pad=0.02)
    cbar2.set_label('Density', fontsize=12, fontweight='bold')
    
    # Mark peak
    ax2.scatter(peak_R0, peak_sigma,
               s=300, c='blue', marker='+',
               linewidths=3, zorder=10,
               label=f'Peak (R0={peak_R0:.3f}, σ={peak_sigma:.3f})')
    ax2.annotate(f'R0={peak_R0:.3f}\nσ={peak_sigma:.3f}',
                xy=(peak_R0, peak_sigma),
                xytext=(15, 15), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    # Mark true values
    if true_R0 is not None and true_sigma is not None:
        ax2.scatter(true_R0, true_sigma,
                   s=300, c='red', marker='*',
                   edgecolors='black', linewidths=1.5,
                   zorder=11, label=f'True (R0={true_R0}, σ={true_sigma})')
        ax2.annotate(f'True\nR0={true_R0}\nσ={true_sigma}',
                    xy=(true_R0, true_sigma),
                    xytext=(-60, 15), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    ax2.set_xlabel('R0', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Sigma', fontsize=13, fontweight='bold')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_title(f'Contour (KDE bw={bw_method})\n({n_selected:,} samples)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, framealpha=0.9)
    
    # Overall title
    fig.suptitle(f'{title} - {percentile}% percentile',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_file = f"{save_path}{save_filename}"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {save_file}")
    plt.close()
    
    return peak_R0, peak_sigma


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":

    # True values from filename
    TRUE_R0 = 1.5
    TRUE_SIGMA = 0.8

    for percentile, filename in [
        (1.0,  'posterior_scatter_contour_1percentile.png'),
        (0.1,  'posterior_scatter_contour_0p1percentile.png'),
        (0.01, 'posterior_scatter_contour_0p01percentile.png'),
    ]:
        print(f"\n{'='*70}")
        print(f"PERCENTILE: {percentile}%")
        print(f"{'='*70}")
        
        # Energy distance
        ed = calculate_energy_distance_abc(
            file_dists='../../experimental_data/from_260528/dists_sigma0p8_R01p5_recal.csv',
            file_R0='../../experimental_data/from_260528/R0.csv',
            file_sigma='../../experimental_data/from_260528/sigma.csv',
            percentile=percentile,
            n_subsample=1000
        )
        print(f"\n✅ Energy distance: {ed:.6f}")
        
        # Plot
        peak_R0, peak_sigma = plot_energy_distance(
            file_dists='../../experimental_data/from_260528/dists_sigma0p8_R01p5_recal.csv',
            file_R0='../../experimental_data/from_260528/R0.csv',
            file_sigma='../../experimental_data/from_260528/sigma.csv',
            percentile=percentile,
            true_R0=TRUE_R0,        # ← True R0 from filename
            true_sigma=TRUE_SIGMA,  # ← True sigma from filename
            xlim=(1, 4),
            ylim=(0.2, 1.0),
            bw_method='silverman',
            title="ABC Posterior",
            save_path='../../figures/from_260528/ppc/sigma0p8/R01p5/',
            save_filename=filename
        )
        print(f"✅ Peak: R0={peak_R0:.4f}, sigma={peak_sigma:.4f}")
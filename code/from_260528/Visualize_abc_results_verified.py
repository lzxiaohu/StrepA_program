import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_dots_multiple_percentiles(file_dists='dists.csv', 
                                   file_R0='R0.csv', 
                                   file_sigma='sigma.csv',
                                   percentiles=[100, 80, 60, 40, 20, 10],
                                   true_R0=None,
                                   true_sigma=None,
                                   title="R0 vs sigma - Multiple Percentiles",
                                   save_path='../../experimental_data/from_260312/',
                                   xlim=(1, 8),
                                   ylim=(0.2, 1.0)):
    """
    Create 6 subplots showing R0 vs sigma for different distance percentiles.
    
    IMPORTANT: percentile=10 means "keep the 10% CLOSEST to the standard point"
               i.e., select samples where distance <= 10th percentile of distances
    """
    # Load data
    distances = pd.read_csv(file_dists, header=None).values.ravel()
    data_R0 = pd.read_csv(file_R0, header=None).values.ravel()
    data_sigma = pd.read_csv(file_sigma, header=None).values.ravel()
    
    print(f"Loaded {len(distances)} samples")
    print(f"Distance range: [{distances.min():.6f}, {distances.max():.6f}]")
    print(f"Distance mean: {distances.mean():.6f}")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Plot for each percentile
    for idx, percentile in enumerate(percentiles):
        ax = axes[idx]
        
        # Calculate threshold - keep samples with distance <= this threshold
        # Lower percentile = smaller threshold = closer to standard point
        if percentile == 100:
            selected_indices = np.arange(len(distances))
            threshold = distances.max()
        else:
            threshold = np.percentile(distances, percentile)
            selected_indices = np.where(distances <= threshold)[0]
        
        n_selected = len(selected_indices)
        
        print(f"\nPercentile {percentile}%:")
        print(f"  Threshold: {threshold:.6f}")
        print(f"  Selected: {n_selected:,} samples ({100*n_selected/len(distances):.2f}%)")
        
        # Select samples
        selected_R0 = data_R0[selected_indices]
        selected_sigma = data_sigma[selected_indices]
        selected_dist = distances[selected_indices]
        
        print(f"  Distance range of selected: [{selected_dist.min():.6f}, {selected_dist.max():.6f}]")
        print(f"  Distance mean of selected: {selected_dist.mean():.6f}")
        
        # Scatter plot
        ax.scatter(selected_R0, selected_sigma, alpha=0.5, s=20, c='blue', 
                  label=f'Selected samples')
        
        # Mark true values
        if true_R0 is not None and true_sigma is not None:
            ax.scatter(true_R0, true_sigma, s=200, c='red', marker='*',
                      edgecolors='black', linewidths=2, label='True value', zorder=5)
        
        # Set fixed ranges
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Labels
        ax.set_xlabel('R0', fontsize=11)
        ax.set_ylabel('sigma', fontsize=11)
        ax.set_title(
            f'{percentile}% closest to standard\n'
            f'({n_selected:,} samples, dist ≤ {threshold:.4f})',
            fontsize=12, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    import os
    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_path}{title.replace(' ', '_').replace('-', '')}.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved to {save_file}")
    plt.close()


def verify_percentile_logic(file_dists, percentiles=[10, 5, 1]):
    """
    Verify that percentile logic is working correctly.
    Lower percentiles should give SMALLER distance thresholds.
    """
    distances = pd.read_csv(file_dists, header=None).values.ravel()
    
    print("="*70)
    print("PERCENTILE LOGIC VERIFICATION")
    print("="*70)
    print(f"Total samples: {len(distances):,}")
    print(f"Distance range: [{distances.min():.6f}, {distances.max():.6f}]")
    print("\nExpected: Lower percentile → Lower threshold → Closer to standard point")
    print("-"*70)
    
    for p in percentiles:
        threshold = np.percentile(distances, p)
        n_selected = (distances <= threshold).sum()
        
        print(f"\nPercentile {p}%:")
        print(f"  Threshold: {threshold:.6f}")
        print(f"  Samples selected: {n_selected:,} ({100*n_selected/len(distances):.2f}%)")
        print(f"  ✓ Correct!" if n_selected < len(distances) else "")
    
    # Check ordering
    print("\n" + "="*70)
    print("THRESHOLD ORDERING CHECK:")
    print("="*70)
    thresholds = [np.percentile(distances, p) for p in percentiles]
    is_increasing = all(thresholds[i] < thresholds[i+1] for i in range(len(thresholds)-1))
    
    if is_increasing:
        print("❌ ERROR: Thresholds are INCREASING (should be DECREASING)")
        print("   This means higher percentiles give tighter selection (WRONG!)")
    else:
        is_decreasing = all(thresholds[i] > thresholds[i+1] for i in range(len(thresholds)-1))
        if is_decreasing:
            print("✅ CORRECT: Thresholds are DECREASING")
            print("   Lower percentile → Lower threshold → Closer points")
        else:
            print("⚠️  Thresholds are neither increasing nor decreasing")
    
    print("\nThresholds:", [f"{t:.6f}" for t in thresholds])
    print("="*70)


# ============================================================================
# EXAMPLE USAGE WITH VERIFICATION
# ============================================================================

if __name__ == "__main__":
    
    # First, verify the percentile logic
    print("STEP 1: Verifying percentile logic...")
    verify_percentile_logic(
        file_dists='../../experimental_data/from_260430/dists_R02p5_recal.csv',
        percentiles=[10, 5, 2, 1, 0.5, 0.1]
    )
    
    # Then generate plots
    print("\n\nSTEP 2: Generating plots...")
    
    plot_dots_multiple_percentiles(
        file_dists='../../experimental_data/from_260430/dists_R02p5_recal.csv',
        file_R0='../../experimental_data/from_260430/R0.csv',
        file_sigma='../../experimental_data/from_260430/sigma.csv',
        percentiles=[10, 5, 4, 3, 2, 1],
        true_R0=2.5,
        true_sigma=0.8,
        title="ABC Posterior - Top Percentiles",
        save_path="../../figures/from_260430/ppc/sigma0p8/R02p5/",
        xlim=(1, 8),
        ylim=(0.2, 1.0)
    )
    
    plot_dots_multiple_percentiles(
        file_dists='../../experimental_data/from_260430/dists_R02p5_recal.csv',
        file_R0='../../experimental_data/from_260430/R0.csv',
        file_sigma='../../experimental_data/from_260430/sigma.csv',
        percentiles=[2, 1, 0.5, 0.4, 0.2, 0.1],
        true_R0=2.5,
        true_sigma=0.8,
        title="ABC Posterior - Tightest Selection",
        save_path="../../figures/from_260430/ppc/sigma0p8/R02p5/",
        xlim=(1, 8),
        ylim=(0.2, 1.0)
    )
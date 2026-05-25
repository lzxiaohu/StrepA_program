# display_selected_samples.py
# Show detailed information about selected samples at a given percentile

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import summary_stats_elms_260305 as ss

def display_selected_samples(
    file_dists='../../experimental_data/from_260430/dists_observations_recal.csv',
    file_R0='../../experimental_data/from_260430/R0.csv',
    file_sigma='../../experimental_data/from_260430/sigma.csv',
    simulation_banks_dir='../../experimental_data/from_260430/simulation_banks',
    valid_indices_file='../../experimental_data/from_260430/valid_indices.csv',
    percentile=0.01,
    n_display=10,
    save_path='../../experimental_data/from_260430/ppc/observations/'
):
    """
    Display detailed information about samples selected at a given percentile.
    Now with proper index mapping support!
    """
    
    print("="*70)
    print(f"DISPLAYING SAMPLES AT {percentile}% PERCENTILE")
    print("="*70)
    
    # Load mapping (clean → original indices)
    print(f"\nLoading index mapping...")
    try:
        valid_indices = np.loadtxt(valid_indices_file, dtype=int)
        print(f"✓ Loaded mapping: {len(valid_indices):,} clean → original indices")
        has_mapping = True
    except FileNotFoundError:
        print(f"⚠️  Warning: Mapping file not found at {valid_indices_file}")
        print(f"   Assuming clean indices = original indices")
        valid_indices = None
        has_mapping = False
    
    # Load data
    print("\nLoading data...")
    distances = pd.read_csv(file_dists, header=None).values.ravel()
    R0_array = pd.read_csv(file_R0, header=None).values.ravel()
    sigma_array = pd.read_csv(file_sigma, header=None).values.ravel()
    
    print(f"✓ Loaded {len(distances):,} samples")
    
    # Select samples at percentile
    threshold = np.percentile(distances, percentile)
    selected_indices = np.where(distances <= threshold)[0]
    n_selected = len(selected_indices)
    
    print(f"\nPercentile {percentile}%:")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Selected: {n_selected:,} samples ({100*n_selected/len(distances):.3f}%)")
    
    # Get selected data
    selected_distances = distances[selected_indices]
    selected_R0 = R0_array[selected_indices]
    selected_sigma = sigma_array[selected_indices]
    
    # Sort by distance (closest first)
    sort_idx = np.argsort(selected_distances)
    
    clean_indices_sorted = selected_indices[sort_idx]
    selected_distances_sorted = selected_distances[sort_idx]
    selected_R0_sorted = selected_R0[sort_idx]
    selected_sigma_sorted = selected_sigma[sort_idx]
    
    # Map to original indices if mapping available
    if has_mapping:
        original_indices_sorted = valid_indices[clean_indices_sorted]
    else:
        original_indices_sorted = clean_indices_sorted
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'rank': np.arange(1, n_selected + 1),
        'clean_id': clean_indices_sorted,
        'original_id': original_indices_sorted,
        'distance': selected_distances_sorted,
        'R0': selected_R0_sorted,
        'sigma': selected_sigma_sorted
    })
    
    # Display top samples
    print(f"\n{'='*70}")
    print(f"TOP {min(n_display, n_selected)} CLOSEST SAMPLES:")
    print(f"{'='*70}")
    if has_mapping:
        print("clean_id    = index in R0.csv/sigma.csv/distances (499,665 samples)")
        print("original_id = index in simulation_bank_*.h5 files (500,000 samples)\n")
    print(summary_df.head(n_display).to_string(index=False))
    
    # Statistics
    print(f"\n{'='*70}")
    print("STATISTICS OF SELECTED SAMPLES:")
    print(f"{'='*70}")
    print(f"Distance range: [{selected_distances_sorted.min():.6f}, {selected_distances_sorted.max():.6f}]")
    print(f"Distance mean:  {selected_distances_sorted.mean():.6f}")
    print(f"\nR0:")
    print(f"  Range: [{selected_R0_sorted.min():.3f}, {selected_R0_sorted.max():.3f}]")
    print(f"  Mean:  {selected_R0_sorted.mean():.3f} ± {selected_R0_sorted.std():.3f}")
    print(f"\nSigma:")
    print(f"  Range: [{selected_sigma_sorted.min():.3f}, {selected_sigma_sorted.max():.3f}]")
    print(f"  Mean:  {selected_sigma_sorted.mean():.3f} ± {selected_sigma_sorted.std():.3f}")
    
    # Save summary to CSV
    output_file = f"{save_path}selected_samples_p{percentile}.csv"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved summary to: {output_file}")
    
    return summary_df
def load_and_plot_simulations(
    sample_ids,
    simulation_banks_dir='../../experimental_data/from_260430/simulation_banks',
    summary_df=None,
    n_plot=5,
    save_path='../../figures/from_260430/ppc/observations/'
):
    """
    Load and plot simulation matrices for specific sample IDs.
    
    Parameters:
    -----------
    sample_ids : array-like
        List of sample IDs to load
    simulation_banks_dir : str
        Directory containing simulation bank files
    summary_df : DataFrame, optional
        DataFrame with sample information (for titles)
    n_plot : int
        Number of simulations to plot
    save_path : str
        Directory to save figures
    """
    
    print(f"\n{'='*70}")
    print(f"LOADING AND PLOTTING SIMULATION MATRICES")
    print(f"{'='*70}")
    
    # Find all simulation bank files
    sim_files = sorted(Path(simulation_banks_dir).glob('simulation_bank_part_*.h5'))
    
    if len(sim_files) == 0:
        print(f"❌ No simulation files found in {simulation_banks_dir}")
        return
    
    print(f"Found {len(sim_files)} simulation bank files")
    
    # Determine file structure (samples per file)
    with h5py.File(sim_files[0], 'r') as f:
        samples_per_file = len(f['R0'])
    
    print(f"Samples per file: {samples_per_file:,}")
    
    # Plot top n_plot simulations
    n_to_plot = min(n_plot, len(sample_ids))
    
    for i in range(n_to_plot):
        sample_id = sample_ids[i]
        
        # Determine which file contains this sample
        file_idx = sample_id // samples_per_file
        local_idx = sample_id % samples_per_file
        
        if file_idx >= len(sim_files):
            print(f"⚠️  Sample {sample_id} is out of range")
            continue
        
        # Load simulation
        sim_file = sim_files[file_idx]
        
        with h5py.File(sim_file, 'r') as f:
            simulation = f['simulations'][local_idx]  # Shape: (40, 23)
            R0_val = f['R0'][local_idx]
            sigma_val = f['sigma'][local_idx]
        
        # Get distance if summary_df provided
        if summary_df is not None:
            row = summary_df[summary_df['sample_id'] == sample_id]
            if not row.empty:
                distance = row['distance'].values[0]
                rank = row['rank'].values[0]
            else:
                distance = None
                rank = i + 1
        else:
            distance = None
            rank = i + 1
        
        # Plot simulation matrix
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(simulation, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        # Labels
        ax.set_xlabel('Time Point', fontsize=12, fontweight='bold')
        ax.set_ylabel('Strain', fontsize=12, fontweight='bold')
        
        title = f'Rank #{rank}: Sample ID {sample_id}\n'
        title += f'R0={R0_val:.3f}, σ={sigma_val:.3f}'
        if distance is not None:
            title += f', Distance={distance:.6f}'
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Prevalence')
        cbar.ax.tick_params(labelsize=10)
        
        # Grid
        ax.set_xticks(np.arange(23))
        ax.set_yticks(np.arange(0, 40, 5))
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save
        Path(save_path).mkdir(parents=True, exist_ok=True)
        save_file = f"{save_path}simulation_rank{rank:03d}_id{sample_id}.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_file}")
        plt.close()
        
        # Print matrix info
        print(f"\n{'='*70}")
        print(f"RANK #{rank}: SAMPLE ID {sample_id}")
        print(f"{'='*70}")
        print(f"File: {sim_file.name}")
        print(f"Local index: {local_idx}")
        print(f"R0: {R0_val:.6f}")
        print(f"Sigma: {sigma_val:.6f}")
        if distance is not None:
            print(f"Distance: {distance:.6f}")
        print(f"Matrix shape: {simulation.shape}")
        print(f"Prevalence range: [{simulation.min():.1f}, {simulation.max():.1f}]")
        print(f"Total infections: {simulation.sum():.0f}")
        print(f"Non-zero entries: {(simulation > 0).sum()} / {simulation.size}")


def plot_comparison_grid(
    sample_ids,
    simulation_banks_dir='../../experimental_data/from_260430/simulation_banks',
    summary_df=None,
    n_cols=3,
    save_path='../../figures/from_260430/ppc/observations/'
):
    """
    Create a grid plot comparing multiple simulations.
    
    Parameters:
    -----------
    sample_ids : array-like
        List of sample IDs to compare
    n_cols : int
        Number of columns in grid
    """
    
    n_samples = len(sample_ids)
    n_rows = int(np.ceil(n_samples / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_samples == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    
    # Find simulation files
    sim_files = sorted(Path(simulation_banks_dir).glob('simulation_bank_part_*.h5'))
    
    with h5py.File(sim_files[0], 'r') as f:
        samples_per_file = len(f['R0'])
    
    for i, sample_id in enumerate(sample_ids):
        ax = axes[i]
        
        # Load simulation
        file_idx = sample_id // samples_per_file
        local_idx = sample_id % samples_per_file
        
        with h5py.File(sim_files[file_idx], 'r') as f:
            simulation = f['simulations'][local_idx]
            R0_val = f['R0'][local_idx]
            sigma_val = f['sigma'][local_idx]
        
        # Plot
        im = ax.imshow(simulation, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        # Title
        if summary_df is not None:
            row = summary_df[summary_df['sample_id'] == sample_id]
            if not row.empty:
                rank = row['rank'].values[0]
                dist = row['distance'].values[0]
                ax.set_title(f'#{rank}: R0={R0_val:.2f}, σ={sigma_val:.2f}\ndist={dist:.5f}',
                           fontsize=10, fontweight='bold')
            else:
                ax.set_title(f'ID {sample_id}: R0={R0_val:.2f}, σ={sigma_val:.2f}',
                           fontsize=10, fontweight='bold')
        else:
            ax.set_title(f'ID {sample_id}: R0={R0_val:.2f}, σ={sigma_val:.2f}',
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Time', fontsize=9)
        ax.set_ylabel('Strain', fontsize=9)
        
        # Colorbar for each subplot
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide extra subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_file = f"{save_path}comparison_grid_top{n_samples}.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison grid: {save_file}")
    plt.close()


def plot_samples_5x2(
    sample_ids,
    simulation_banks_dir='../../experimental_data/from_260430/simulation_banks',
    file_R0='../../experimental_data/from_260430/R0.csv',
    file_sigma='../../experimental_data/from_260430/sigma.csv',
    file_dists='../../experimental_data/from_260430/dists_observations_recal.csv',
    title="Selected Simulations",
    save_path='../../figures/from_260430/ppc/observations/',
    save_filename='specific_samples_5x2.png'
):
    """
    Plot exactly 10 specific sample IDs in a 5x2 grid.
    
    Parameters:
    -----------
    sample_ids : list or array
        List of 10 sample IDs to plot
    simulation_banks_dir : str
        Directory containing simulation bank HDF5 files
    file_R0 : str
        CSV file with R0 values
    file_sigma : str
        CSV file with sigma values
    file_dists : str, optional
        CSV file with distances (optional, for displaying distance)
    title : str
        Overall figure title
    save_path : str
        Directory to save figure
    save_filename : str
        Filename for saved figure
    """
    
    # Ensure exactly 10 samples
    sample_ids = np.array(sample_ids)
    if len(sample_ids) != 10:
        print(f"⚠️  Warning: Expected 10 sample IDs, got {len(sample_ids)}")
        print(f"    Using first 10 or padding with None")
        if len(sample_ids) > 10:
            sample_ids = sample_ids[:10]
        else:
            sample_ids = np.pad(sample_ids, (0, 10 - len(sample_ids)), 
                               constant_values=-1)
    
    print("="*70)
    print(f"PLOTTING 10 SPECIFIC SAMPLES IN 5×2 GRID")
    print("="*70)
    print(f"Sample IDs: {sample_ids}")
    
    # Load parameters
    R0_array = pd.read_csv(file_R0, header=None).values.ravel()
    sigma_array = pd.read_csv(file_sigma, header=None).values.ravel()
    
    # Load distances if available
    try:
        distances = pd.read_csv(file_dists, header=None).values.ravel()
        has_distances = True
    except:
        has_distances = False
        print("Note: No distance file provided, skipping distance display")
    
    # Find simulation bank files
    sim_files = sorted(Path(simulation_banks_dir).glob('simulation_bank_part_*.h5'))
    
    if len(sim_files) == 0:
        print(f"❌ Error: No simulation files found in {simulation_banks_dir}")
        return
    
    # Determine samples per file
    with h5py.File(sim_files[0], 'r') as f:
        samples_per_file = len(f['R0'])
    
    print(f"Found {len(sim_files)} simulation bank files")
    print(f"Samples per file: {samples_per_file:,}")
    
    # Create 5x2 grid
    fig, axes = plt.subplots(5, 2, figsize=(14, 20))
    axes = axes.ravel()
    
    for i, sample_id in enumerate(sample_ids):
        ax = axes[i]
        
        if sample_id < 0 or sample_id >= len(R0_array):
            # Invalid sample ID, leave blank
            ax.axis('off')
            ax.text(0.5, 0.5, f'Invalid ID: {sample_id}', 
                   ha='center', va='center', fontsize=12)
            continue
        
        # Determine which file contains this sample
        file_idx = int(sample_id // samples_per_file)
        local_idx = int(sample_id % samples_per_file)
        
        if file_idx >= len(sim_files):
            ax.axis('off')
            ax.text(0.5, 0.5, f'ID {sample_id} out of range', 
                   ha='center', va='center', fontsize=12)
            continue
        
        # Load simulation matrix
        sim_file = sim_files[file_idx]
        
        try:
            with h5py.File(sim_file, 'r') as f:
                simulation = f['simulations'][local_idx]  # Shape: (40, 23)
        except Exception as e:
            ax.axis('off')
            ax.text(0.5, 0.5, f'Error loading\nID {sample_id}', 
                   ha='center', va='center', fontsize=12)
            print(f"Error loading sample {sample_id}: {e}")
            continue
        
        # Get parameters
        R0_val = R0_array[sample_id]
        sigma_val = sigma_array[sample_id]
        
        if has_distances:
            dist_val = distances[sample_id]
        
        # Plot heatmap
        im = ax.imshow(simulation, aspect='auto', cmap='YlOrRd', 
                      interpolation='nearest', vmin=0)
        
        # Title with parameters
        title_text = f'ID {sample_id}: R0={R0_val:.3f}, σ={sigma_val:.3f}'
        if has_distances:
            title_text += f'\nDist={dist_val:.5f}'
        
        ax.set_title(title_text, fontsize=11, fontweight='bold')
        
        # Labels
        ax.set_xlabel('Time', fontsize=9)
        ax.set_ylabel('Strain', fontsize=9)
        ax.tick_params(labelsize=8)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Prevalence', fontsize=8)
        
        print(f"  ✓ Plotted sample {sample_id}: R0={R0_val:.3f}, σ={sigma_val:.3f}")
    
    # Overall title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_file = f"{save_path}{save_filename}"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {save_file}")
    plt.close()


def plot_matrices_scatter_5x2(
    sample_ids,
    simulation_banks_dir='../../experimental_data/from_260430/simulation_banks',
    file_R0='../../experimental_data/from_260430/R0.csv',
    file_sigma='../../experimental_data/from_260430/sigma.csv',
    file_dists='../../experimental_data/from_260430/dists_observations_recal.csv',
    title="Simulation Matrices - Scatter View",
    save_path='../../figures/from_260430/ppc/observations/',
    save_filename='matrices_scatter_5x2.png',
    vmin=0,      # Minimum value for color scale
    vmax=80      # Maximum value for color scale
):
    """
    Plot 10 simulation matrices as scatter plots in a 5x2 grid.
    Each point represents a non-zero prevalence value at (time, strain).
    
    Parameters:
    -----------
    sample_ids : list or array
        List of 10 sample IDs to plot
    simulation_banks_dir : str
        Directory containing simulation bank HDF5 files
    file_R0 : str
        CSV file with R0 values
    file_sigma : str
        CSV file with sigma values
    file_dists : str, optional
        CSV file with distances
    title : str
        Overall figure title
    save_path : str
        Directory to save figure
    save_filename : str
        Filename for saved figure
    vmin : float
        Minimum value for color scale (default: 0)
    vmax : float
        Maximum value for color scale (default: 80)
    """
    
    # Ensure exactly 10 samples
    sample_ids = np.array(sample_ids)
    if len(sample_ids) != 10:
        print(f"⚠️  Warning: Expected 10 sample IDs, got {len(sample_ids)}")
        if len(sample_ids) > 10:
            sample_ids = sample_ids[:10]
        else:
            sample_ids = np.pad(sample_ids, (0, 10 - len(sample_ids)), 
                               constant_values=-1)
    
    print("="*70)
    print(f"PLOTTING 10 SIMULATION MATRICES AS SCATTER PLOTS (5×2 GRID)")
    print("="*70)
    print(f"Sample IDs: {sample_ids}")
    
    # Load parameters
    R0_array = pd.read_csv(file_R0, header=None).values.ravel()
    sigma_array = pd.read_csv(file_sigma, header=None).values.ravel()
    
    # Load distances if available
    try:
        distances = pd.read_csv(file_dists, header=None).values.ravel()
        has_distances = True
    except:
        has_distances = False
    
    # Find simulation bank files
    sim_files = sorted(Path(simulation_banks_dir).glob('simulation_bank_part_*.h5'))
    
    if len(sim_files) == 0:
        print(f"❌ Error: No simulation files found in {simulation_banks_dir}")
        return
    
    # Determine samples per file
    with h5py.File(sim_files[0], 'r') as f:
        samples_per_file = len(f['R0'])
    
    print(f"Found {len(sim_files)} simulation bank files")
    
    # Create 5x2 grid
    fig, axes = plt.subplots(5, 2, figsize=(14, 20))
    axes = axes.ravel()
    
    for i, sample_id in enumerate(sample_ids):
        ax = axes[i]
        
        if sample_id < 0 or sample_id >= len(R0_array):
            # Invalid sample ID
            ax.axis('off')
            ax.text(0.5, 0.5, f'Invalid ID: {sample_id}', 
                   ha='center', va='center', fontsize=12)
            continue
        
        # Determine which file contains this sample
        file_idx = int(sample_id // samples_per_file)
        local_idx = int(sample_id % samples_per_file)
        
        if file_idx >= len(sim_files):
            ax.axis('off')
            ax.text(0.5, 0.5, f'ID {sample_id} out of range', 
                   ha='center', va='center', fontsize=12)
            continue
        
        # Load simulation matrix
        sim_file = sim_files[file_idx]
        
        try:
            with h5py.File(sim_file, 'r') as f:
                simulation = f['simulations'][local_idx]  # Shape: (40, 23)
        except Exception as e:
            ax.axis('off')
            ax.text(0.5, 0.5, f'Error loading\nID {sample_id}', 
                   ha='center', va='center', fontsize=12)
            print(f"Error loading sample {sample_id}: {e}")
            continue
        
        # Get parameters
        R0_val = R0_array[sample_id]
        sigma_val = sigma_array[sample_id]
        
        if has_distances:
            dist_val = distances[sample_id]
        
        # Convert matrix to scatter points
        # For each non-zero value, create a point at (time, strain)
        strains, times = np.where(simulation > 0)  # Get indices where value > 0
        prevalences = simulation[strains, times]  # Get the actual values
        
        # Create scatter plot
        scatter = ax.scatter(times, strains, 
                           c=prevalences, 
                           s=prevalences * 10,  # Size proportional to prevalence
                           cmap='YlOrRd', 
                           alpha=0.7,
                           edgecolors='black',
                           linewidths=0.3,
                           vmin=vmin,  # Fixed color scale
                           vmax=vmax)  # Fixed color scale
        
        # Labels and limits
        ax.set_xlabel('Time Point', fontsize=10)
        ax.set_ylabel('Strain', fontsize=10)
        ax.set_xlim(-0.5, 22.5)
        ax.set_ylim(-0.5, 39.5)
        
        # Title with parameters
        title_text = f'ID {sample_id}: R0={R0_val:.3f}, σ={sigma_val:.3f}'
        if has_distances:
            title_text += f'\nDist={dist_val:.5f}'
        
        ax.set_title(title_text, fontsize=10, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Prevalence', fontsize=8)
        
        # Print info
        n_nonzero = (simulation > 0).sum()
        print(f"  ✓ Sample {sample_id}: R0={R0_val:.3f}, σ={sigma_val:.3f}, "
              f"{n_nonzero} non-zero points")
    
    # Overall title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_file = f"{save_path}{save_filename}"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {save_file}")
    plt.close()
 
def plot_matrices_scatter_single(
    sample_ids,
    simulation_banks_dir='../../experimental_data/from_260430/simulation_banks',
    file_R0='../../experimental_data/from_260430/R0.csv',
    file_sigma='../../experimental_data/from_260430/sigma.csv',
    title="Overlaid Simulation Matrices",
    save_path='../../figures/from_260430/ppc/observations/',
    save_filename='matrices_scatter_overlay.png'
):
    """
    Plot all simulation matrices overlaid on a single plot with different colors.
    
    Parameters:
    -----------
    sample_ids : list or array
        List of sample IDs to overlay
    """
    
    print("="*70)
    print(f"PLOTTING {len(sample_ids)} MATRICES OVERLAID")
    print("="*70)
    
    # Load parameters
    R0_array = pd.read_csv(file_R0, header=None).values.ravel()
    sigma_array = pd.read_csv(file_sigma, header=None).values.ravel()
    
    # Find simulation files
    sim_files = sorted(Path(simulation_banks_dir).glob('simulation_bank_part_*.h5'))
    
    with h5py.File(sim_files[0], 'r') as f:
        samples_per_file = len(f['R0'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Colors for different samples
    colors = plt.cm.tab10(np.linspace(0, 1, len(sample_ids)))
    
    for idx, sample_id in enumerate(sample_ids):
        if sample_id < 0 or sample_id >= len(R0_array):
            continue
        
        # Load simulation
        file_idx = int(sample_id // samples_per_file)
        local_idx = int(sample_id % samples_per_file)
        
        with h5py.File(sim_files[file_idx], 'r') as f:
            simulation = f['simulations'][local_idx]
        
        R0_val = R0_array[sample_id]
        sigma_val = sigma_array[sample_id]
        
        # Get scatter points
        strains, times = np.where(simulation > 0)
        prevalences = simulation[strains, times]
        
        # Plot
        ax.scatter(times, strains, 
                  c=[colors[idx]], 
                  s=prevalences * 8,
                  alpha=0.6,
                  label=f'ID {sample_id} (R0={R0_val:.2f}, σ={sigma_val:.2f})',
                  edgecolors='black',
                  linewidths=0.3)
    
    # Labels
    ax.set_xlabel('Time Point', fontsize=14, fontweight='bold')
    ax.set_ylabel('Strain', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, 22.5)
    ax.set_ylim(-0.5, 39.5)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_file = f"{save_path}{save_filename}"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {save_file}")
    plt.close()


def plot_observation_heatmap(
    observation_matrix=None,
    title="Observed Data - Heatmap",
    save_path='../../figures/from_260430/ppc/observations/',
    save_filename='observation_heatmap.png'
):
    """
    Plot observation matrix as a heatmap.
    
    Parameters:
    -----------
    observation_matrix : np.ndarray, optional
        40x23 observation matrix. If None, uses the default observations.
    title : str
        Figure title
    save_path : str
        Directory to save figure
    save_filename : str
        Filename for saved figure
    """
    
    if observation_matrix is None:
        observation_matrix = observations
    
    print("="*70)
    print("PLOTTING OBSERVATION MATRIX - HEATMAP")
    print("="*70)
    print(f"Matrix shape: {observation_matrix.shape}")
    print(f"Total infections: {observation_matrix.sum():.0f}")
    print(f"Non-zero entries: {(observation_matrix > 0).sum()}")
    print(f"Max prevalence: {observation_matrix.max():.0f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Heatmap
    im = ax.imshow(observation_matrix, aspect='auto', cmap='YlOrRd', 
                   interpolation='nearest', vmin=0)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Prevalence (Number of Cases)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Labels
    ax.set_xlabel('Time Point', fontsize=14, fontweight='bold')
    ax.set_ylabel('Strain', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    # Ticks
    ax.set_xticks(np.arange(23))
    ax.set_yticks(np.arange(0, 40, 5))
    ax.tick_params(labelsize=11)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add text box with statistics
    stats_text = (f"Total infections: {observation_matrix.sum():.0f}\n"
                 f"Non-zero entries: {(observation_matrix > 0).sum()}\n"
                 f"Max prevalence: {observation_matrix.max():.0f}")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_file = f"{save_path}{save_filename}"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {save_file}")
    plt.close()
 
 
def plot_observation_scatter(
    observation_matrix=None,
    title="Observed Data - Scatter Plot",
    save_path='../../figures/from_260430/ppc/observations/',
    save_filename='observation_scatter.png',
    vmin=0,   # Minimum value for color scale
    vmax=80   # Maximum value for color scale
):
    """
    Plot observation matrix as a scatter plot.
    Points are placed at (time, strain) with size and color representing prevalence.
    
    Parameters:
    -----------
    observation_matrix : np.ndarray, optional
        40x23 observation matrix. If None, uses the default observations.
    title : str
        Figure title
    save_path : str
        Directory to save figure
    save_filename : str
        Filename for saved figure
    vmin : float
        Minimum value for color scale (default: 0)
    vmax : float
        Maximum value for color scale (default: 80)
    """
    
    if observation_matrix is None:
        observation_matrix = observations
    
    print("="*70)
    print("PLOTTING OBSERVATION MATRIX - SCATTER")
    print("="*70)
    print(f"Matrix shape: {observation_matrix.shape}")
    
    # Get scatter points (only non-zero values)
    strains, times = np.where(observation_matrix > 0)
    prevalences = observation_matrix[strains, times]
    
    n_points = len(strains)
    print(f"Non-zero points: {n_points}")
    print(f"Total infections: {prevalences.sum():.0f}")
    print(f"Prevalence range: [{prevalences.min():.0f}, {prevalences.max():.0f}]")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Scatter plot
    scatter = ax.scatter(
        times, 
        strains, 
        c=prevalences,
        s=prevalences * 30,  # Size proportional to prevalence
        cmap='YlOrRd',
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5,
        vmin=vmin,  # Fixed color scale
        vmax=vmax   # Fixed color scale
    )
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Prevalence (Number of Cases)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Labels
    ax.set_xlabel('Time Point', fontsize=14, fontweight='bold')
    ax.set_ylabel('Strain', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    # Limits
    ax.set_xlim(-0.5, 22.5)
    ax.set_ylim(-0.5, 39.5)
    
    # Ticks
    ax.set_xticks(np.arange(23))
    ax.set_yticks(np.arange(0, 40, 5))
    ax.tick_params(labelsize=11)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add text box with statistics
    stats_text = (f"Non-zero points: {n_points}\n"
                 f"Total infections: {prevalences.sum():.0f}\n"
                 f"Max prevalence: {prevalences.max():.0f}\n"
                 f"Color scale: [{vmin}, {vmax}]")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_file = f"{save_path}{save_filename}"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {save_file}")
    plt.close()
 
 
 
def plot_observation_both(
    observation_matrix=None,
    save_path='../../figures/from_260430/ppc/observations/'
):
    """
    Generate both heatmap and scatter plot versions of the observation matrix.
    
    Parameters:
    -----------
    observation_matrix : np.ndarray, optional
        40x23 observation matrix. If None, uses the default observations.
    save_path : str
        Directory to save figures
    """
    
    if observation_matrix is None:
        observation_matrix = observations
    
    # Version 1: Heatmap
    plot_observation_heatmap(
        observation_matrix=observation_matrix,
        title="Observed Data - Heatmap",
        save_path=save_path,
        save_filename='observation_heatmap.png'
    )
    
    # Version 2: Scatter
    plot_observation_scatter(
        observation_matrix=observation_matrix,
        title="Observed Data - Scatter Plot",
        save_path=save_path,
        save_filename='observation_scatter.png'
    )
    
    print("\n" + "="*70)
    print("✅ BOTH VERSIONS GENERATED!")
    print("="*70)
 

def calculate_summary_stats_single(matrix):
    """
    Calculate summary statistics for a single 40x23 matrix.
    
    Parameters:
    -----------
    matrix : np.ndarray
        40x23 simulation matrix
    
    Returns:
    --------
    dict : Summary statistics
    """
    avg_prev = ss.avg_prev_numpy(matrix)
    var_prev = np.sqrt(ss.var_prev_numpy(matrix))
    avg_npmi = ss.avg_npmi_numpy(matrix)
    div_all_isolates = ss.div_all_isolates_numpy(matrix)
    
    return {
        'avg_prev_obs': avg_prev,
        'var_prev_obs': var_prev,
        'avg_npmi_obs': avg_npmi,
        'div_all_isolates_obs': div_all_isolates
    }
 
 
def calculate_summary_stats_comparison(
    sample_ids,
    observation_matrix=None,
    simulation_banks_dir='../../experimental_data/from_260430/simulation_banks',
    file_R0='../../experimental_data/from_260430/R0.csv',
    file_sigma='../../experimental_data/from_260430/sigma.csv',
    file_dists='../../experimental_data/from_260430/dists_observations_recal.csv',
    save_path='../../experimental_data/from_260430/ppc/observations/',
    save_filename='summary_stats_comparison.csv'
):
    """
    Calculate summary statistics for observation and selected samples.
    
    Parameters:
    -----------
    sample_ids : list or array
        List of sample IDs to analyze
    observation_matrix : np.ndarray, optional
        Observation matrix (40x23). If None, uses default.
    simulation_banks_dir : str
        Directory with simulation bank files
    file_R0 : str
        R0 values CSV
    file_sigma : str
        Sigma values CSV
    file_dists : str
        Distances CSV
    save_path : str
        Where to save results
    save_filename : str
        Output CSV filename
    
    Returns:
    --------
    pd.DataFrame : Comparison table
    """
    
    if observation_matrix is None:
        observation_matrix = observations
    
    print("="*70)
    print("CALCULATING SUMMARY STATISTICS COMPARISON")
    print("="*70)
    
    # Calculate observation statistics
    print("\nCalculating observation statistics...")
    obs_stats = calculate_summary_stats_single(observation_matrix)
    
    print("Observation summary statistics:")
    for key, value in obs_stats.items():
        print(f"  {key:25s}: {value:.6f}")
    
    # Load parameters
    R0_array = pd.read_csv(file_R0, header=None).values.ravel()
    sigma_array = pd.read_csv(file_sigma, header=None).values.ravel()
    distances = pd.read_csv(file_dists, header=None).values.ravel()
    
    # Find simulation files
    sim_files = sorted(Path(simulation_banks_dir).glob('simulation_bank_part_*.h5'))
    
    with h5py.File(sim_files[0], 'r') as f:
        samples_per_file = len(f['R0'])
    
    # Calculate statistics for each sample
    results = []
    
    # Add observation as first row
    results.append({
        'ID': 'Observation',
        'R0': np.nan,
        'sigma': np.nan,
        'distance': np.nan,
        'avg_prev_obs': obs_stats['avg_prev_obs'],
        'var_prev_obs': obs_stats['var_prev_obs'],
        'avg_npmi_obs': obs_stats['avg_npmi_obs'],
        'div_all_isolates_obs': obs_stats['div_all_isolates_obs']
    })
    
    print(f"\nCalculating statistics for {len(sample_ids)} samples...")
    
    for idx, sample_id in enumerate(sample_ids, 1):
        if sample_id < 0 or sample_id >= len(R0_array):
            print(f"  Skipping invalid sample ID: {sample_id}")
            continue
        
        # Load simulation
        file_idx = int(sample_id // samples_per_file)
        local_idx = int(sample_id % samples_per_file)
        
        try:
            with h5py.File(sim_files[file_idx], 'r') as f:
                simulation = f['simulations'][local_idx]
        except Exception as e:
            print(f"  Error loading sample {sample_id}: {e}")
            continue
        
        # Get parameters
        R0_val = R0_array[sample_id]
        sigma_val = sigma_array[sample_id]
        dist_val = distances[sample_id]
        
        # Calculate statistics
        sim_stats = calculate_summary_stats_single(simulation)
        
        results.append({
            'ID': int(sample_id),
            'R0': R0_val,
            'sigma': sigma_val,
            'distance': dist_val,
            'avg_prev_obs': sim_stats['avg_prev_obs'],
            'var_prev_obs': sim_stats['var_prev_obs'],
            'avg_npmi_obs': sim_stats['avg_npmi_obs'],
            'div_all_isolates_obs': sim_stats['div_all_isolates_obs']
        })
        
        print(f"  ✓ Sample {sample_id}: R0={R0_val:.3f}, σ={sigma_val:.3f}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate differences from observation
    for col in ['avg_prev_obs', 'var_prev_obs', 'avg_npmi_obs', 'div_all_isolates_obs']:
        obs_value = obs_stats[col]
        df[f'{col}_diff'] = df[col] - obs_value
        df[f'{col}_diff_pct'] = ((df[col] - obs_value) / obs_value * 100) if obs_value != 0 else np.nan
    
    # Display comparison
    print("\n" + "="*70)
    print("SUMMARY STATISTICS COMPARISON TABLE")
    print("="*70)
    
    # Show main statistics
    display_cols = ['ID', 'R0', 'sigma', 'distance', 
                   'avg_prev_obs', 'var_prev_obs', 'avg_npmi_obs', 'div_all_isolates_obs']
    print("\n" + df[display_cols].to_string(index=False))
    
    # Show differences
    print("\n" + "="*70)
    print("DIFFERENCES FROM OBSERVATION")
    print("="*70)
    
    diff_cols = ['ID'] + [col for col in df.columns if '_diff' in col and '_pct' not in col]
    print("\n" + df[diff_cols].to_string(index=False))
    
    # Save to CSV
    Path(save_path).mkdir(parents=True, exist_ok=True)
    output_file = f"{save_path}{save_filename}"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved full comparison to: {output_file}")
    
    return df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Step 1: Get selected samples at 0.01% percentile
    summary_df = display_selected_samples(
        file_dists='../../experimental_data/from_260430/dists_observations_recal.csv',
        file_R0='../../experimental_data/from_260430/R0.csv',
        file_sigma='../../experimental_data/from_260430/sigma.csv',
        simulation_banks_dir='../../experimental_data/from_260430/simulation_banks',
        valid_indices_file='../../experimental_data/from_260430/valid_indices.csv',  # ← Add this!
        percentile=0.01,
        n_display=20,
        save_path='../../experimental_data/from_260430/ppc/observations/'
    )
    
    # Step 2: Plot top 5 simulations individually
    top_sample_ids = summary_df['sample_id'].values[:5]
    
    load_and_plot_simulations(
        sample_ids=top_sample_ids,
        simulation_banks_dir='../../experimental_data/from_260430/simulation_banks',
        summary_df=summary_df,
        n_plot=5,
        save_path='../../figures/from_260430/ppc/observations/'
    )
    
    # Step 3: Create comparison grid of top 9 samples
    top_9_ids = summary_df['sample_id'].values[:9]
    
    plot_comparison_grid(
        sample_ids=top_9_ids,
        simulation_banks_dir='../../experimental_data/from_260430/simulation_banks',
        summary_df=summary_df,
        n_cols=3,
        save_path='../../figures/from_260430/ppc/observations/'
    )
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)

    # my_sample_ids = [125516, 479082, 114075, 80433, 302555, 
    #                  95410, 162873, 360406, 395833, 60592]
    my_sample_ids = [125516, 479082, 114075, 80433, 302555, 
                     95410, 162873, 360406, 395833, 60592]


    
    
    plot_samples_5x2(
        sample_ids=my_sample_ids,
        simulation_banks_dir='../../experimental_data/from_260430/simulation_banks',
        file_R0='../../experimental_data/from_260430/R0.csv',
        file_sigma='../../experimental_data/from_260430/sigma.csv',
        file_dists='../../experimental_data/from_260430/dists_observations_recal.csv',
        title="Selected Simulations - 5×2 Grid",
        save_path='../../figures/from_260430/ppc/observations/',
        save_filename='my_selected_samples.png'
    )

    plot_matrices_scatter_5x2(
        sample_ids=my_sample_ids,
        simulation_banks_dir='../../experimental_data/from_260430/simulation_banks',
        title="Specified 10 Simulations - Scatter View",
        save_filename='matrices_scatter_grid.png',
        vmin=0,   # Color scale minimum
        vmax=80   # Color scale maximum)
    )

    observations = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 1, 5, 2, 4, 2, 0, 0, 3, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 5, 3, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 13, 7, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 3, 1, 3, 0, 2, 1, 2, 0, 1],
    [0, 0, 0, 0, 0, 0, 2, 3, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 4, 1, 1, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 2, 3, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 4, 1, 0, 1, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 3, 1, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 2, 2, 5, 4, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0]
])

plot_observation_both(
        observation_matrix=observations,
        save_path='../../figures/from_260430/ppc/observations/'
    )

df = calculate_summary_stats_comparison(
        sample_ids=my_sample_ids,
        observation_matrix=observations,
        save_filename='summary_stats_comparison_top10.csv'
    )
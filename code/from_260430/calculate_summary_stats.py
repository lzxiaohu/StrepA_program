# calculate_summary_statistics_clean.py
# Calculate summary statistics and REMOVE rows with NaN values

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import summary_stats_elms_260305 as ss
import time


def summary_stats(series_2d):
    """Calculate summary statistics for a single simulation."""
    avg_prev_obs = ss.avg_prev_numpy(series_2d)
    var_prev_obs = np.sqrt(ss.var_prev_numpy(series_2d))
    avg_npmi_obs = ss.avg_npmi_numpy(series_2d)
    div_all_isolates_obs = ss.div_all_isolates_numpy(series_2d)
    
    return np.array([avg_prev_obs, var_prev_obs, avg_npmi_obs, div_all_isolates_obs], float)


def process_single_simulation(args):
    """Worker function for parallel processing."""
    simulation, sample_id = args
    stats = summary_stats(simulation)
    return sample_id, stats


def calculate_summary_stats_clean(
    input_dir='../../experimental_data/from_260430/simulation_banks',
    output_file='../../experimental_data/from_260430/all_summary_statistics_clean.h5',
    n_jobs=40
):
    """
    Calculate summary statistics and REMOVE rows with NaN values.
    This ensures all data is clean and ready for analysis.
    """
    
    input_path = Path(input_dir)
    files = sorted(input_path.glob('simulation_bank_part_*.h5'))
    
    # Count total samples
    total_samples = 0
    for filepath in files:
        with h5py.File(filepath, 'r') as f:
            total_samples += len(f['R0'])
    
    print("="*70)
    print("CALCULATING SUMMARY STATISTICS → CLEAN DATA (NaN REMOVAL)")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Input files: {len(files)}")
    print(f"Total samples: {total_samples:,}")
    print(f"Output file: {output_file}")
    print(f"Parallel workers: {n_jobs}")
    print("="*70)
    
    # Pre-allocate arrays
    print(f"\nAllocating arrays for {total_samples:,} samples...")
    all_summary_stats = np.zeros((total_samples, 4), dtype=np.float32)
    all_R0 = np.zeros(total_samples, dtype=np.float32)
    all_sigma = np.zeros(total_samples, dtype=np.float32)
    
    start_time = time.time()
    current_idx = 0
    
    # Process each file
    for file_num, filepath in enumerate(files, 1):
        print(f"\n{'='*70}")
        print(f"FILE {file_num}/{len(files)}: {filepath.name}")
        print(f"{'='*70}")
        
        with h5py.File(filepath, 'r') as f:
            n_samples = len(f['R0'])
            print(f"Samples in this file: {n_samples:,}")
            print(f"Global index: {current_idx:,} to {current_idx + n_samples - 1:,}")
            
            # Load data
            simulations = f['simulations'][:]
            R0_array = f['R0'][:]
            sigma_array = f['sigma'][:]
            
            # Store parameters
            all_R0[current_idx:current_idx + n_samples] = R0_array
            all_sigma[current_idx:current_idx + n_samples] = sigma_array
            
            # Prepare arguments
            args_list = [(simulations[i], current_idx + i) for i in range(n_samples)]
            
            # Calculate statistics in parallel
            print(f"Calculating summary statistics with {n_jobs} workers...")
            with mp.Pool(n_jobs) as pool:
                results = list(tqdm(
                    pool.imap(process_single_simulation, args_list),
                    total=len(args_list),
                    desc="  Processing",
                    ncols=80
                ))
            
            # Store results
            for sample_id, stats in results:
                all_summary_stats[sample_id] = stats
            
            current_idx += n_samples
            
            # Progress update
            elapsed = time.time() - start_time
            rate = current_idx / elapsed
            remaining = total_samples - current_idx
            eta = remaining / rate if rate > 0 else 0
            
            print(f"Progress: {current_idx:,}/{total_samples:,} ({100*current_idx/total_samples:.1f}%)")
            print(f"Rate: {rate:.1f} samples/sec")
            print(f"ETA: {eta/60:.1f} minutes")
    
    # REMOVE ROWS WITH NaN
    print(f"\n{'='*70}")
    print("CHECKING FOR NaN VALUES...")
    print(f"{'='*70}")
    
    # Find rows with any NaN
    has_nan = np.any(np.isnan(all_summary_stats), axis=1)
    n_nan_rows = has_nan.sum()
    n_valid_rows = (~has_nan).sum()
    
    print(f"Original samples: {total_samples:,}")
    print(f"Rows with NaN:    {n_nan_rows:,} ({100*n_nan_rows/total_samples:.3f}%)")
    print(f"Valid rows:       {n_valid_rows:,} ({100*n_valid_rows/total_samples:.3f}%)")
    
    if n_nan_rows > 0:
        print(f"\n⚠️  Removing {n_nan_rows:,} rows with NaN values...")
        print(f"   Keeping {n_valid_rows:,} clean samples")
        
        # Keep only valid rows (remove entire row if any NaN)
        all_summary_stats_clean = all_summary_stats[~has_nan]
        all_R0_clean = all_R0[~has_nan]
        all_sigma_clean = all_sigma[~has_nan]
        final_n_samples = n_valid_rows
    else:
        print(f"\n✓ No NaN values found! All data is clean.")
        all_summary_stats_clean = all_summary_stats
        all_R0_clean = all_R0
        all_sigma_clean = all_sigma
        final_n_samples = total_samples
    
    # Save clean data
    print(f"\n{'='*70}")
    print("SAVING CLEAN DATA TO FILE...")
    print(f"{'='*70}")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_file, 'w') as f:
        # Save clean summary statistics
        f.create_dataset(
            'summary_stats',
            data=all_summary_stats_clean,
            compression='gzip',
            compression_opts=6,
            chunks=(min(10000, final_n_samples), 4)
        )
        
        # Save clean parameters
        f.create_dataset(
            'R0',
            data=all_R0_clean,
            compression='gzip',
            compression_opts=6
        )
        
        f.create_dataset(
            'sigma',
            data=all_sigma_clean,
            compression='gzip',
            compression_opts=6
        )
        
        # Add metadata
        f.attrs['n_samples'] = final_n_samples
        f.attrs['n_samples_original'] = total_samples
        f.attrs['n_samples_removed'] = n_nan_rows
        f.attrs['removal_rate'] = float(n_nan_rows) / total_samples
        f.attrs['n_statistics'] = 4
        f.attrs['columns'] = ['avg_prev_obs', 'var_prev_obs', 'avg_npmi_obs', 'div_all_isolates_obs']
        f.attrs['R0_range'] = [1.0, 8.0]
        f.attrs['sigma_range'] = [0.2, 1.0]
        f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['data_quality'] = 'clean (NaN rows removed)'
    
    # Final summary
    total_time = time.time() - start_time
    file_size_mb = output_path.stat().st_size / (1024**2)
    
    print(f"\n{'='*70}")
    print("✅ SUMMARY STATISTICS COMPLETE (CLEAN DATA)!")
    print(f"{'='*70}")
    print(f"Original samples:  {total_samples:,}")
    print(f"Removed (NaN):     {n_nan_rows:,} ({100*n_nan_rows/total_samples:.3f}%)")
    print(f"Final samples:     {final_n_samples:,} ({100*n_valid_rows/total_samples:.3f}%)")
    print(f"Total time:        {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average rate:      {total_samples/total_time:.1f} samples/sec")
    print(f"Output file:       {output_file}")
    print(f"File size:         {file_size_mb:.2f} MB ({file_size_mb/1024:.2f} GB)")
    print(f"{'='*70}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    calculate_summary_stats_clean(
        input_dir='../../experimental_data/from_260430/simulation_banks',
        output_file='../../experimental_data/from_260430/all_summary_statistics_clean.h5',
        n_jobs=40
    )
    
    print("\n" + "="*70)
    print("DATA IS NOW CLEAN AND READY FOR ANALYSIS!")
    print("="*70)
    print("""
Next steps:
1. Use 'all_summary_statistics_clean.h5' for your ABC analysis
2. All rows with NaN have been removed
3. R0, sigma, and summary_stats are all aligned
4. No NaN values remain in the dataset
    """)
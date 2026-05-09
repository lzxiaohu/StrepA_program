# calculate_summary_statistics_single_file.py
# Calculate summary statistics for all 500k simulations and save to ONE file

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import summary_stats_elms_260305 as ss
import time


def summary_stats(series_2d):
    """
    Calculate summary statistics for a single simulation.

    Parameters:
    -----------
    series_2d : np.ndarray
        40x23 matrix (strains x timepoints)

    Returns:
    --------
    np.ndarray : [avg_prev_obs, var_prev_obs, avg_npmi_obs, div_all_isolates_obs]
    """
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


def calculate_summary_stats_single_file(
        input_dir='../../experimental_data/from_260430/simulation_banks',
        output_file='../../experimental_data/from_260430/all_summary_statistics.h5',
        n_jobs=24
):
    """
    Calculate summary statistics for all simulations and save to ONE file.

    Parameters:
    -----------
    input_dir : str
        Directory containing simulation_bank_part_*.h5 files
    output_file : str
        Output HDF5 file path (single file for all stats)
    n_jobs : int
        Number of parallel workers
    """

    input_path = Path(input_dir)
    files = sorted(input_path.glob('simulation_bank_part_*.h5'))

    # Count total samples
    total_samples = 0
    for filepath in files:
        with h5py.File(filepath, 'r') as f:
            total_samples += len(f['R0'])

    print("=" * 70)
    print("CALCULATING SUMMARY STATISTICS → SINGLE FILE")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Input files: {len(files)}")
    print(f"Total samples: {total_samples:,}")
    print(f"Output file: {output_file}")
    print(f"Parallel workers: {n_jobs}")
    print("=" * 70)

    # Pre-allocate arrays for ALL samples
    print(f"\nAllocating arrays for {total_samples:,} samples...")
    all_summary_stats = np.zeros((total_samples, 4), dtype=np.float32)
    all_R0 = np.zeros(total_samples, dtype=np.float32)
    all_sigma = np.zeros(total_samples, dtype=np.float32)

    start_time = time.time()
    current_idx = 0

    # Process each file
    for file_num, filepath in enumerate(files, 1):
        print(f"\n{'=' * 70}")
        print(f"FILE {file_num}/{len(files)}: {filepath.name}")
        print(f"{'=' * 70}")

        with h5py.File(filepath, 'r') as f:
            n_samples = len(f['R0'])
            print(f"Samples in this file: {n_samples:,}")
            print(f"Global index: {current_idx:,} to {current_idx + n_samples - 1:,}")

            # Load data from this file
            simulations = f['simulations'][:]
            R0_array = f['R0'][:]
            sigma_array = f['sigma'][:]

            # Store parameters
            all_R0[current_idx:current_idx + n_samples] = R0_array
            all_sigma[current_idx:current_idx + n_samples] = sigma_array

            # Prepare arguments for parallel processing
            args_list = [(simulations[i], current_idx + i) for i in range(n_samples)]

            # Calculate summary statistics in parallel
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

            print(f"Progress: {current_idx:,}/{total_samples:,} ({100 * current_idx / total_samples:.1f}%)")
            print(f"Rate: {rate:.1f} samples/sec")
            print(f"ETA: {eta / 60:.1f} minutes")

    # Save everything to ONE file
    print(f"\n{'=' * 70}")
    print("SAVING TO SINGLE FILE...")
    print(f"{'=' * 70}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, 'w') as f:
        # Save summary statistics
        f.create_dataset(
            'summary_stats',
            data=all_summary_stats,
            compression='gzip',
            compression_opts=6,
            chunks=(min(10000, total_samples), 4)
        )

        # Save parameters
        f.create_dataset(
            'R0',
            data=all_R0,
            compression='gzip',
            compression_opts=6
        )

        f.create_dataset(
            'sigma',
            data=all_sigma,
            compression='gzip',
            compression_opts=6
        )

        # Add metadata
        f.attrs['n_samples'] = total_samples
        f.attrs['n_statistics'] = 4
        f.attrs['columns'] = ['avg_prev_obs', 'var_prev_obs', 'avg_npmi_obs', 'div_all_isolates_obs']
        f.attrs['R0_range'] = [1.0, 8.0]
        f.attrs['sigma_range'] = [0.2, 1.0]
        f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

    # Final summary
    total_time = time.time() - start_time
    file_size_mb = output_path.stat().st_size / (1024 ** 2)

    print(f"\n{'=' * 70}")
    print("✅ SUMMARY STATISTICS COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Total samples: {total_samples:,}")
    print(f"Total time: {total_time / 60:.1f} minutes ({total_time / 3600:.2f} hours)")
    print(f"Average rate: {total_samples / total_time:.1f} samples/sec")
    print(f"Output file: {output_file}")
    print(f"File size: {file_size_mb:.2f} MB ({file_size_mb / 1024:.2f} GB)")
    print(f"{'=' * 70}")


def load_summary_stats_single_file(filepath='../../experimental_data/from_260430/all_summary_statistics.h5'):
    """
    Load summary statistics from the single output file.

    Returns:
    --------
    dict : {
        'summary_stats': (500000, 4) array,
        'R0': (500000,) array,
        'sigma': (500000,) array,
        'columns': list of column names
    }
    """
    print(f"Loading summary statistics from: {filepath}")

    with h5py.File(filepath, 'r') as f:
        data = {
            'summary_stats': f['summary_stats'][:],
            'R0': f['R0'][:],
            'sigma': f['sigma'][:],
            'columns': list(f.attrs['columns']),
            'n_samples': f.attrs['n_samples']
        }

    print(f"✓ Loaded {data['n_samples']:,} samples")
    print(f"  Summary stats shape: {data['summary_stats'].shape}")
    print(f"  Columns: {data['columns']}")

    return data


def get_summary_stats_info(filepath='../../experimental_data/from_260430/all_summary_statistics.h5'):
    """Print information about the summary statistics file."""

    with h5py.File(filepath, 'r') as f:
        print("=" * 70)
        print("SUMMARY STATISTICS FILE INFO")
        print("=" * 70)
        print(f"File: {filepath}")
        print(f"File size: {Path(filepath).stat().st_size / (1024 ** 2):.2f} MB")
        print(f"\nDatasets:")
        print(f"  summary_stats: {f['summary_stats'].shape} {f['summary_stats'].dtype}")
        print(f"  R0: {f['R0'].shape} {f['R0'].dtype}")
        print(f"  sigma: {f['sigma'].shape} {f['sigma'].dtype}")
        print(f"\nMetadata:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        print("=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Calculate and save to single file
    calculate_summary_stats_single_file(
        input_dir='../../experimental_data/from_260430/simulation_banks',
        output_file='../../experimental_data/from_260430/all_summary_statistics.h5',
        n_jobs=24
    )

    # Show how to use
    print("\n" + "=" * 70)
    print("HOW TO LOAD THE RESULTS:")
    print("=" * 70)
    print("""
# Load all summary statistics
from calculate_summary_statistics_single_file import load_summary_stats_single_file

data = load_summary_stats_single_file()

summary_stats = data['summary_stats']  # (500000, 4) array
R0 = data['R0']                        # (500000,)
sigma = data['sigma']                  # (500000,)
columns = data['columns']              # ['avg_prev_obs', 'var_prev_obs', ...]

# Access individual columns
avg_prev = summary_stats[:, 0]
var_prev = summary_stats[:, 1]
avg_npmi = summary_stats[:, 2]
div_all_isolates = summary_stats[:, 3]

# Or use column names
col_idx = columns.index('avg_prev_obs')
avg_prev = summary_stats[:, col_idx]
    """)
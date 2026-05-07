# file name: load_simulation_banks_optimized.py
# Load simulation bank results from BATCHED ARRAY storage (optimized HDF5)

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm


def load_simulation_bank_sample(output_dir, sample_id):
    """
    Load a single simulation and its parameters from batched arrays.
    
    Parameters:
    -----------
    output_dir : str
        Directory containing simulation_bank_part_*.h5 files
    sample_id : int
        Sample ID to load (0 to 499,999)
    
    Returns:
    --------
    dict : {
        'simulation': np.ndarray (40, 23),
        'R0': float,
        'sigma': float
    }
    
    Example:
    --------
    >>> result = load_simulation_bank_sample('simulation_banks', 12345)
    >>> SSPrev_selected = result['simulation']  # 40x23 matrix
    >>> R0 = result['R0']
    >>> sigma = result['sigma']
    """
    files = sorted(Path(output_dir).glob('simulation_bank_part_*.h5'))
    
    for filepath in files:
        with h5py.File(filepath, 'r') as f:
            start = f.attrs['start_sample']
            end = f.attrs['end_sample']
            
            if start <= sample_id < end:
                # Calculate index within this file
                local_idx = sample_id - start
                
                # Load from batched arrays
                simulation = f['simulations'][local_idx]
                R0 = f['R0'][local_idx]
                sigma = f['sigma'][local_idx]
                
                return {
                    'simulation': simulation,
                    'R0': float(R0),
                    'sigma': float(sigma)
                }
    
    raise ValueError(f"Sample {sample_id} not found in {output_dir}")


def load_sample_range(output_dir, start_id, end_id):
    """
    Load a range of samples efficiently from batched arrays.
    MUCH FASTER than loading individually!
    
    Parameters:
    -----------
    output_dir : str
        Directory containing simulation files
    start_id : int
        Starting sample ID (inclusive)
    end_id : int
        Ending sample ID (exclusive)
    
    Returns:
    --------
    dict : {
        'simulations': np.ndarray (n_samples, 40, 23),
        'R0': np.ndarray (n_samples,),
        'sigma': np.ndarray (n_samples,)
    }
    
    Example:
    --------
    >>> data = load_sample_range('simulation_banks', 1000, 2000)
    >>> print(data['simulations'].shape)  # (1000, 40, 23)
    """
    n_samples = end_id - start_id
    files = sorted(Path(output_dir).glob('simulation_bank_part_*.h5'))
    
    # Pre-allocate output arrays
    simulations = np.zeros((n_samples, 40, 23))
    R0_values = np.zeros(n_samples)
    sigma_values = np.zeros(n_samples)
    
    current_idx = 0
    
    for filepath in files:
        with h5py.File(filepath, 'r') as f:
            file_start = f.attrs['start_sample']
            file_end = f.attrs['end_sample']
            
            # Check if this file contains any samples we need
            if file_end <= start_id or file_start >= end_id:
                continue
            
            # Calculate overlap
            overlap_start = max(start_id, file_start)
            overlap_end = min(end_id, file_end)
            
            # Local indices within this file
            local_start = overlap_start - file_start
            local_end = overlap_end - file_start
            
            # Number of samples to copy
            n_to_copy = overlap_end - overlap_start
            
            # Load slice from batched arrays (FAST!)
            simulations[current_idx:current_idx+n_to_copy] = f['simulations'][local_start:local_end]
            R0_values[current_idx:current_idx+n_to_copy] = f['R0'][local_start:local_end]
            sigma_values[current_idx:current_idx+n_to_copy] = f['sigma'][local_start:local_end]
            
            current_idx += n_to_copy
    
    return {
        'simulations': simulations,
        'R0': R0_values,
        'sigma': sigma_values
    }


def load_all_simulations(output_dir, max_samples=None):
    """
    Load all simulations efficiently from batched arrays.
    MUCH FASTER than individual dataset loading!
    
    Parameters:
    -----------
    output_dir : str
        Directory containing simulation files
    max_samples : int, optional
        Maximum number of samples to load (for testing)
    
    Returns:
    --------
    dict : {
        'simulations': np.ndarray (n_samples, 40, 23),
        'R0': np.ndarray (n_samples,),
        'sigma': np.ndarray (n_samples,)
    }
    
    Example:
    --------
    >>> data = load_all_simulations('simulation_banks')
    >>> SSPrev_all = data['simulations']  # (500000, 40, 23)
    >>> R0_all = data['R0']  # (500000,)
    >>> sigma_all = data['sigma']  # (500000,)
    """
    files = sorted(Path(output_dir).glob('simulation_bank_part_*.h5'))
    
    # Get total number of samples
    total_samples = 0
    for filepath in files:
        with h5py.File(filepath, 'r') as f:
            total_samples += f.attrs['n_samples']
    
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)
    
    print(f"Loading {total_samples:,} samples from batched arrays...")
    print(f"Memory required: ~{total_samples * 40 * 23 * 4 / 1024**3:.2f} GB (float32)")
    
    # Pre-allocate arrays
    simulations = np.zeros((total_samples, 40, 23), dtype=np.float32)
    R0_values = np.zeros(total_samples, dtype=np.float32)
    sigma_values = np.zeros(total_samples, dtype=np.float32)
    
    # Load data file by file (batched reads are FAST!)
    loaded_count = 0
    for filepath in tqdm(files, desc="Loading files"):
        with h5py.File(filepath, 'r') as f:
            n_samples = f.attrs['n_samples']
            
            # Calculate how many to load from this file
            n_to_load = min(n_samples, total_samples - loaded_count)
            
            if n_to_load == 0:
                break
            
            # Load entire batch at once (FAST!)
            simulations[loaded_count:loaded_count+n_to_load] = f['simulations'][:n_to_load]
            R0_values[loaded_count:loaded_count+n_to_load] = f['R0'][:n_to_load]
            sigma_values[loaded_count:loaded_count+n_to_load] = f['sigma'][:n_to_load]
            
            loaded_count += n_to_load
            
            if loaded_count >= total_samples:
                break
    
    return {
        'simulations': simulations,
        'R0': R0_values,
        'sigma': sigma_values
    }


def load_file_data(filepath):
    """
    Load all data from a single file.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to simulation_bank_part_*.h5 file
    
    Returns:
    --------
    dict : {
        'simulations': np.ndarray,
        'R0': np.ndarray,
        'sigma': np.ndarray,
        'start_sample': int,
        'end_sample': int
    }
    """
    with h5py.File(filepath, 'r') as f:
        return {
            'simulations': f['simulations'][:],
            'R0': f['R0'][:],
            'sigma': f['sigma'][:],
            'start_sample': f.attrs['start_sample'],
            'end_sample': f.attrs['end_sample']
        }


def get_simulation_bank_info(output_dir):
    """
    Get information about simulation bank.
    
    Returns:
    --------
    dict : Information about the simulation bank
    """
    files = sorted(Path(output_dir).glob('simulation_bank_part_*.h5'))
    
    total_samples = 0
    total_size_mb = 0
    file_info = []
    
    for filepath in files:
        size_mb = filepath.stat().st_size / (1024**2)
        total_size_mb += size_mb
        
        with h5py.File(filepath, 'r') as f:
            n_samples = f.attrs['n_samples']
            start = f.attrs['start_sample']
            end = f.attrs['end_sample']
            R0_range = f.attrs['R0_range']
            sigma_range = f.attrs['sigma_range']
            
            total_samples += n_samples
            
            file_info.append({
                'filename': filepath.name,
                'samples': n_samples,
                'start': start,
                'end': end,
                'size_mb': size_mb
            })
    
    return {
        'total_files': len(files),
        'total_samples': total_samples,
        'total_size_mb': total_size_mb,
        'R0_range': R0_range,
        'sigma_range': sigma_range,
        'files': file_info
    }


def print_simulation_bank_summary(output_dir):
    """
    Print summary of simulation bank.
    """
    info = get_simulation_bank_info(output_dir)
    
    print("="*70)
    print("SIMULATION BANK SUMMARY (Batched Array Storage)")
    print("="*70)
    print(f"Directory: {output_dir}")
    print(f"Total files: {info['total_files']}")
    print(f"Total samples: {info['total_samples']:,}")
    print(f"Total size: {info['total_size_mb']:.2f} MB ({info['total_size_mb']/1024:.2f} GB)")
    print(f"R0 range: {info['R0_range']}")
    print(f"Sigma range: {info['sigma_range']}")
    print(f"\nStorage format: 3 batched arrays per file (optimized HDF5)")
    print(f"  - simulations: (n_samples, 40, 23)")
    print(f"  - R0: (n_samples,)")
    print(f"  - sigma: (n_samples,)")
    print(f"\nFiles:")
    print("-"*70)
    
    for f in info['files']:
        print(f"{f['filename']:35s} | {f['size_mb']:6.2f} MB | "
              f"Samples {f['start']:7,} to {f['end']-1:7,} ({f['samples']:6,} total)")
    
    print("="*70)


def export_to_numpy(output_dir, output_file='simulation_bank_all.npz'):
    """
    Export entire simulation bank to a single .npz file.
    
    Parameters:
    -----------
    output_dir : str
        Directory containing simulation files
    output_file : str
        Output .npz filename
    
    Example:
    --------
    >>> export_to_numpy('simulation_banks', 'all_data.npz')
    >>> # Later load with:
    >>> data = np.load('all_data.npz')
    >>> simulations = data['simulations']
    >>> R0 = data['R0']
    >>> sigma = data['sigma']
    """
    data = load_all_simulations(output_dir)
    
    print(f"Saving to {output_file}...")
    np.savez_compressed(
        output_file,
        simulations=data['simulations'],
        R0=data['R0'],
        sigma=data['sigma']
    )
    
    # Check file size
    size_mb = Path(output_file).stat().st_size / (1024**2)
    print(f"Saved! File size: {size_mb:.2f} MB")


# Example usage
if __name__ == "__main__":
    
    # Print summary
    print_simulation_bank_summary('../../experimental_data/from_260430/simulation_banks')
    
    # Load a single sample
    print("\nLoading sample 0...")
    result = load_simulation_bank_sample('../../experimental_data/from_260430/simulation_banks', 0)
    print(f"Simulation shape: {result['simulation'].shape}")
    print(f"simulation: {result['simulation']}")
    print(f"R0: {result['R0']:.4f}")
    print(f"Sigma: {result['sigma']:.4f}")
    
    # Load a range (FAST with batched arrays!)
    print("\nLoading samples 100-200...")
    data = load_sample_range('../../experimental_data/from_260430/simulation_banks', 100, 200)
    print(f"Simulations shape: {data['simulations'].shape}")
    print(f"R0 range: [{data['R0'].min():.2f}, {data['R0'].max():.2f}]")
    print(f"Sigma range: [{data['sigma'].min():.2f}, {data['sigma'].max():.2f}]")
    
    # Load first 100 samples for testing
    print("\nLoading first 100 samples...")
    data = load_all_simulations('../../experimental_data/from_260430/simulation_banks', max_samples=100)
    print(f"Simulations shape: {data['simulations'].shape}")

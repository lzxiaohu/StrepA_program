# test_index_mapping.py
# Verify which index corresponds to which data source

import numpy as np
import pandas as pd
import h5py
from pathlib import Path


def test_index_mapping():
    """
    Test the index mapping hypothesis by checking R0 and sigma values.
    """
    
    print("="*70)
    print("TESTING INDEX MAPPING HYPOTHESIS")
    print("="*70)
    
    # Load clean data from CSV files
    print("\n1. Loading CLEAN data from CSV files (499,665 samples)...")
    R0_csv = pd.read_csv('../../experimental_data/from_260430/R0.csv', header=None).values.ravel()
    sigma_csv = pd.read_csv('../../experimental_data/from_260430/sigma.csv', header=None).values.ravel()
    
    print(f"   R0.csv: {len(R0_csv):,} samples")
    print(f"   sigma.csv: {len(sigma_csv):,} samples")
    
    # Load mapping if exists
    print("\n2. Loading mapping file (if exists)...")
    try:
        valid_indices = np.loadtxt('../../experimental_data/from_260430/valid_indices.csv', dtype=int)
        print(f"   ✓ valid_indices.csv: {len(valid_indices):,} entries")
        has_mapping = True
    except FileNotFoundError:
        print(f"   ✗ valid_indices.csv NOT FOUND")
        print(f"   We need to create this file first!")
        has_mapping = False
        return
    
    # Load simulation banks
    print("\n3. Loading simulation banks (500,000 samples)...")
    sim_files = sorted(Path('../../experimental_data/from_260430/simulation_banks').glob('simulation_bank_part_*.h5'))
    print(f"   Found {len(sim_files)} files")
    
    with h5py.File(sim_files[0], 'r') as f:
        samples_per_file = len(f['R0'])
    
    print(f"   Samples per file: {samples_per_file:,}")
    
    # Test cases
    print("\n" + "="*70)
    print("TEST CASES")
    print("="*70)
    
    # Test case 1: From your example
    test_cases = [
        {'clean_id': 125516, 'expected_original': 125598, 
         'expected_R0_csv': 1.055655, 'expected_sigma_csv': 0.92125946},
        {'clean_id': 0, 'expected_original': None, 
         'expected_R0_csv': None, 'expected_sigma_csv': None},
        {'clean_id': 100, 'expected_original': None, 
         'expected_R0_csv': None, 'expected_sigma_csv': None},
    ]
    
    for i, test in enumerate(test_cases, 1):
        clean_id = test['clean_id']
        
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}: Clean ID = {clean_id}")
        print(f"{'='*70}")
        
        # Get values from CSV using clean_id
        R0_from_csv = R0_csv[clean_id]
        sigma_from_csv = sigma_csv[clean_id]
        
        print(f"\nFrom CSV files (using clean_id {clean_id}):")
        print(f"  R0    = {R0_from_csv:.6f}")
        print(f"  sigma = {sigma_from_csv:.6f}")
        
        if test['expected_R0_csv'] is not None:
            print(f"  Expected: R0={test['expected_R0_csv']:.6f}, sigma={test['expected_sigma_csv']:.6f}")
            match = (abs(R0_from_csv - test['expected_R0_csv']) < 0.0001 and 
                    abs(sigma_from_csv - test['expected_sigma_csv']) < 0.0001)
            print(f"  {'✓ MATCH!' if match else '✗ NO MATCH'}")
        
        # Map to original_id
        original_id = valid_indices[clean_id]
        print(f"\nMapping: clean_id {clean_id} → original_id {original_id}")
        
        if test['expected_original'] is not None:
            match = (original_id == test['expected_original'])
            print(f"  Expected original_id: {test['expected_original']}")
            print(f"  {'✓ MATCH!' if match else '✗ NO MATCH'}")
        
        # Load from simulation banks using original_id
        file_idx = int(original_id // samples_per_file)
        local_idx = int(original_id % samples_per_file)
        
        print(f"\nFrom simulation banks (using original_id {original_id}):")
        print(f"  File: simulation_bank_part_{file_idx:04d}.h5")
        print(f"  Local index: {local_idx}")
        
        with h5py.File(sim_files[file_idx], 'r') as f:
            R0_from_h5 = f['R0'][local_idx]
            sigma_from_h5 = f['sigma'][local_idx]
            simulation = f['simulations'][local_idx]
        
        print(f"  R0    = {R0_from_h5:.6f}")
        print(f"  sigma = {sigma_from_h5:.6f}")
        print(f"  Total infections in matrix: {simulation.sum():.0f}")
        
        # Check if CSV and HDF5 match
        print(f"\n{'='*70}")
        print("COMPARISON:")
        print(f"{'='*70}")
        print(f"CSV  (clean_id {clean_id:6d}):     R0={R0_from_csv:.6f}, sigma={sigma_from_csv:.6f}")
        print(f"HDF5 (original_id {original_id:6d}): R0={R0_from_h5:.6f}, sigma={sigma_from_h5:.6f}")
        
        # They should be DIFFERENT (because clean_id ≠ original_id after NaN removal)
        r0_diff = abs(R0_from_csv - R0_from_h5)
        sigma_diff = abs(sigma_from_csv - sigma_from_h5)
        
        print(f"\nDifferences:")
        print(f"  R0 difference:    {r0_diff:.6f}")
        print(f"  sigma difference: {sigma_diff:.6f}")
        
        if r0_diff < 0.0001 and sigma_diff < 0.0001:
            print(f"\n  ⚠️  CSV and HDF5 are THE SAME!")
            print(f"      This means clean_id {clean_id} was NOT removed (no NaN)")
        else:
            print(f"\n  ✓ CSV and HDF5 are DIFFERENT (as expected)")
            print(f"      This confirms the mapping is working correctly!")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
The correct workflow is:
1. Start with CLEAN_ID from your CSV files (R0.csv, sigma.csv, distances)
2. Use valid_indices[CLEAN_ID] to get ORIGINAL_ID
3. Load simulation matrix from HDF5 using ORIGINAL_ID
4. Use R0 and sigma from CSV (step 1), NOT from HDF5

Example:
  clean_id = 125516
  R0_csv[125516] = 1.055655 (correct)
  original_id = valid_indices[125516] = 125598
  simulation = h5['simulations'][125598] (correct matrix)
  R0_h5[125598] = 3.335 (IGNORE THIS, it's from removed data)
    """)


if __name__ == "__main__":
    test_index_mapping()
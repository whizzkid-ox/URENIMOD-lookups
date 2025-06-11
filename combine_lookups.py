# Author: Ryo Segawa
# Combine lookup tables for NME (Neuronal Mechanics Excitation)

"""
Combines multiple lookup table files from the NME/lookup_tables directory into a single file.
The script automatically finds the lookup_tables directory relative to its location.

Usage:
    python combine_lookups.py <folder_name>

Arguments:
    folder_name: Name of the subfolder in NME/lookup_tables containing .pkl files to combine

Example:
    python combine_lookups.py unmyelinated_fiber_20250527

The combined file will be saved in NME/lookup_tables/combined/<folder_name>.pkl
"""

import os
import glob
import pickle
import numpy as np
import argparse

def load_lookup(file_path):
    """Load a single lookup table from a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def find_closest_index(array, value):
    """Find the index of the closest value in an array."""
    return np.abs(array - value).argmin()

def get_lookup_values(data, params):
    """
    Get lookup table values for specific parameters.
    params: dictionary with keys 'D0', 'd0', 'L0', 'f', 'A', 'Q' and their values
    """
    # Check parameter ranges and find closest values
    indices = {}
    used_values = {}
    
    for key in ['D0', 'd0', 'L0', 'f', 'A', 'Q']:
        if key not in data['refs']:
            print(f"Warning: {key} not found in refs")
            continue
            
        ref_values = data['refs'][key]
        target_value = params[key]
        idx = find_closest_index(ref_values, target_value)
        actual_value = ref_values[idx]
        
        if not np.isclose(actual_value, target_value, rtol=0.1):
            print(f"\nWarning: Requested {key}={target_value} not found in lookup table")
            print(f"Using closest available value: {key}={actual_value}")
            
        indices[key] = idx
        used_values[key] = actual_value
    
    print("\nUsing parameter values:")
    for key, value in used_values.items():
        if key in ['D0', 'd0', 'L0']:
            print(f"  • {key}: {value:.2e} m")
        elif key == 'f':
            print(f"  • {key}: {value:.2e} Hz")
        elif key == 'A':
            print(f"  • {key}: {value:.2e} Pa")
        else:
            print(f"  • {key}: {value:.2f} C/m²")
    
    # Get all table values at these indices
    results = {}
    # Get order of dimensions from the first table's shape
    first_table = next(iter(data['tables'].values()))
    n_dims = len(first_table.shape)
    
    # Create index tuple matching the table dimensions
    for key, value in data['tables'].items():
        try:
            # For 6D tables (2,1,1,1,1,1), we need idx_tuple to match this shape
            idx_tuple = [0] * n_dims  # Start with zeros for all dimensions
            for i, k in enumerate(['L0', 'D0', 'd0', 'f', 'A', 'Q']):
                if k in indices:
                    idx_tuple[i] = indices[k]
            results[key] = value[tuple(idx_tuple)]
        except Exception as e:
            print(f"Warning: Could not get value for {key}: {e}")
    
    return results

def print_structure(lookup_data):
    """Print the structure of the lookup table."""
    print("\nLookup Table Structure:")
    print("=" * 50)
    
    print("\nReference Values:")
    for key, value in lookup_data['refs'].items():
        print(f"  • {key}: Shape {value.shape}, Range: {value.min()} to {value.max()}")
    
    print("\nTable Values:")
    for key, value in lookup_data['tables'].items():
        print(f"  • {key}: Shape {value.shape}")

def combine_lookups(folder_name):
    """Combine all lookup tables from the specified folder."""
    # Setup paths
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory containing this script
    base_dir = os.path.join(script_dir, 'NME', 'lookup_tables')  # NME is a subdirectory of the script directory
    print(f"\nLooking for lookup tables in: {base_dir}")
    source_dir = os.path.join(base_dir, folder_name)
    combined_dir = os.path.join(base_dir, 'combined')
    os.makedirs(combined_dir, exist_ok=True)
    
    # Get all pkl files
    pkl_files = glob.glob(os.path.join(source_dir, '*.pkl'))
    if not pkl_files:
        raise ValueError(f"No .pkl files found in {source_dir}")
    
    print(f"Found {len(pkl_files)} lookup tables to combine")
    
    # Load the first file to get structure
    combined_data = load_lookup(pkl_files[0])
    refs = combined_data['refs']
    tables = combined_data['tables']
    
    # Process remaining files
    for file_path in pkl_files[1:]:
        data = load_lookup(file_path)
        
        # Update refs
        for key in refs:
            if key in data['refs']:
                refs[key] = np.unique(np.concatenate([refs[key], data['refs'][key]]))
        
        # Update tables
        for key in tables:
            if key in data['tables']:
                tables[key] = np.concatenate([tables[key], data['tables'][key]])
    
    # Save combined lookup table
    output_file = os.path.join(combined_dir, f"{folder_name}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump({'refs': refs, 'tables': tables}, f)
    
    print(f"\nCombined lookup table saved to: {output_file}")
    return {'refs': refs, 'tables': tables}

def main():
    parser = argparse.ArgumentParser(description='Combine lookup tables and extract values')
    parser.add_argument('folder_name', type=str, help='Name of the folder containing lookup tables')
    args = parser.parse_args()
    
    # Combine lookup tables
    combined_data = combine_lookups(args.folder_name)
    print_structure(combined_data)
    
    # Parameter set to find values for
    params = {
        'D0': 1e-6,    # m
        'd0': 1.4e-9,  # m
        'L0': 1e-4,    # m
        'f': 1e6,      # Hz
        'A': 1e6,      # Pa
        'Q': -65.0     # C/m²
    }
    
    print("\nLooking up values for parameters:")
    for key, value in params.items():
        print(f"  • {key}: {value}")
    
    results = get_lookup_values(combined_data, params)
    
    print("\nResults:")
    print("=" * 50)
    
    # Section lengths
    if 'l0' in results:
        print("\nSection Length:")
        print(f"  • l0: {results['l0']*1e6:.2f} µm")
    
    # Membrane properties
    print("\nMembrane Properties:")
    if 'V' in results:
        print(f"  • Membrane potential: {results['V']:.2f} mV")
    if 'Rm' in results:
        print(f"  • Membrane resistance: {results['Rm']/1e6:.2f} MΩ")
    
    # Gating variables
    print("\nGating Variables:")
    for gate in ['m', 'h', 'n', 'l']:
        if gate in results:
            print(f"  • {gate}: {results[gate]:.3f}")
            if f'{gate}_velocity' in results:
                print(f"    - velocity: {results[f'{gate}_velocity']:.2e} s⁻¹")
            if f'alpha_{gate}' in results:
                print(f"    - alpha: {results[f'alpha_{gate}']:.2f} s⁻¹")
            if f'beta_{gate}' in results:
                print(f"    - beta: {results[f'beta_{gate}']:.2f} s⁻¹")
    
    # Ionic currents
    print("\nIonic Currents:")
    for current in ['iNa', 'iKd', 'iLeak']:
        if current in results:
            print(f"  • {current}: {results[current]:.2f} mA/m²")
    
    # Computation time
    if 'tcomp' in results:
        print(f"\nComputation time: {results['tcomp']:.2e} s")

if __name__ == '__main__':
    main()

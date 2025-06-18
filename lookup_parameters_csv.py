# Author: Ryo Segawa (whizznihil.kid@gmail.com)

"""
Generate CSV file with parameter combinations for lookup table generation.
The script automatically saves the output file in a 'params_csv' directory
created next to this script.

Output file:
    <script_location>/params_csv/lookup_parameters_YYYYMMDD.csv
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

def generate_parameters():
    """Generate parameter combinations efficiently using numpy meshgrid."""
    # Define parameter ranges with reasonable step sizes to avoid memory issues
    fiber_length = np.arange(0.4e-3, 10e-3 + 0.1e-3, 0.1e-3)  # 0.4mm to 10mm, step 0.1mm (97 values)
    fixed_fiber_length = np.array([5e-3])  # Fixed at 5mm
    fiber_diameter = np.array([1e-6])                      # 1µm constant (1 value)
    membrane_thickness = np.array([1.4e-9])                # 1.4nm constant (1 value)
    freq = np.arange(1e4, 2e6 + 1e4, 1e4)                 # 10kHz to 2MHz, step 10kHz 
    fixed_freq = np.array([500e3])                         # Fixed at 500kHz
    amp = np.arange(100, 1e6 + 1e4, 1e4)                  # 100Pa to 1MPa, step 10kPa
    fixed_amp = np.array([50e3])                           # Fixed at 50kPa
    charge = np.arange(-97, 50 + 1, 1)                    # -97 to 50 nC/cm² (148 values)

#test
    # fiber_length = np.array([1e-4, 2e-4])  # 0.4mm to 10mm, step 0.1mm
    # fiber_diameter = np.array([1e-6])                      # 1µm constant
    # membrane_thickness = np.array([1.4e-9])                # 1.4nm constant
    # freq = np.array([1e6])                      # 10kHz to 5MHz, step 1Hz
    # amp = np.array([1e6])                     # 100Pa to 1MPa, step 1Pa
    # charge = np.array([-65])                   # -97 to 50 nC/cm²
    
    # Generate three sets of parameter combinations
    print("\nGenerating parameter combinations...")
    
    # Set 1: fiber_length * fiber_diameter * membrane_thickness * fixed_freq * fixed_amp * charge
    print("  Generating set 1: varying fiber_length and charge...")
    grid1 = np.meshgrid(fiber_length, fiber_diameter, membrane_thickness, fixed_freq, fixed_amp, charge, indexing='ij')
    combinations1 = np.column_stack([g.ravel() for g in grid1])
    
    # Set 2: fixed_fiber_length * fiber_diameter * membrane_thickness * freq * fixed_amp * charge
    print("  Generating set 2: varying freq and charge...")
    grid2 = np.meshgrid(fixed_fiber_length, fiber_diameter, membrane_thickness, freq, fixed_amp, charge, indexing='ij')
    combinations2 = np.column_stack([g.ravel() for g in grid2])
    
    # Set 3: fixed_fiber_length * fiber_diameter * membrane_thickness * fixed_freq * amp * charge
    print("  Generating set 3: varying amp and charge...")
    grid3 = np.meshgrid(fixed_fiber_length, fiber_diameter, membrane_thickness, fixed_freq, amp, charge, indexing='ij')
    combinations3 = np.column_stack([g.ravel() for g in grid3])
    
    # Combine all sets
    print("  Combining all parameter sets...")
    all_combinations = np.vstack([combinations1, combinations2, combinations3])
    
    # Convert to DataFrame
    columns = ['fiber_length', 'fiber_diameter', 'membrane_thickness', 'freq', 'amp', 'charge']
    df = pd.DataFrame(all_combinations, columns=columns)
    
    # Remove duplicates if any
    df_unique = df.drop_duplicates()
    print(f"  Removed {len(df) - len(df_unique)} duplicate combinations")
    
    return df_unique

def main():
    # Create output directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'params_csv')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with today's date
    today = datetime.now().strftime("%Y%m%d")
    filename = f"lookup_parameters_{today}.csv"
    output_path = os.path.join(output_dir, filename)
      # Generate parameters and save to CSV
    print("Generating parameter combinations...")
    df = generate_parameters()
    print(f"Generated {len(df)} total parameter combinations")
    print(f"\nParameter ranges:")
    print(f"  • fiber_length: {df['fiber_length'].min():.1e} to {df['fiber_length'].max():.1e} m")
    print(f"  • fiber_diameter: {df['fiber_diameter'].unique()[0]:.1e} m")
    print(f"  • membrane_thickness: {df['membrane_thickness'].unique()[0]:.1e} m")
    print(f"  • frequency: {df['freq'].min():.1e} to {df['freq'].max():.1e} Hz")
    print(f"  • amplitude: {df['amp'].min():.1e} to {df['amp'].max():.1e} Pa")
    print(f"  • charge: {df['charge'].min():.1f} to {df['charge'].max():.1f} nC/cm²")
    
    print(f"\nCombination breakdown:")
    print(f"  • Set 1 (varying fiber_length & charge): {len(df[df['freq'] == 500e3][df['amp'] == 50e3])} combinations")
    print(f"  • Set 2 (varying freq & charge): {len(df[df['fiber_length'] == 5e-3][df['amp'] == 50e3])} combinations")
    print(f"  • Set 3 (varying amp & charge): {len(df[df['fiber_length'] == 5e-3][df['freq'] == 500e3])} combinations")
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nParameter combinations saved to: {output_path}")

if __name__ == '__main__':
    main()

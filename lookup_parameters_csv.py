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
    fiber_length = np.array(1e-3)  # 0.4mm to 10mm, step 0.1mm (97 values)
    fiber_diameter = np.array([1e-6])                      # 1µm constant (1 value)
    membrane_thickness = np.array([1.4e-9])                # 1.4nm constant (1 value)
    freq = np.arange(1e4, 2e6 + 1e4, 1e4)                 # 10kHz to 2MHz, step 10kHz 
    amp = np.arange(100, 1e6 + 1e4, 1e4)                  # 100Pa to 1MPa, step 10kPa
    charge = np.arange(-97, 50 + 1, 1)                    # -97 to 50 nC/cm² (148 values)

#test
    # fiber_length = np.array([1e-4, 2e-4])  # 0.4mm to 10mm, step 0.1mm
    # fiber_diameter = np.array([1e-6])                      # 1µm constant
    # membrane_thickness = np.array([1.4e-9])                # 1.4nm constant
    # freq = np.array([1e6])                      # 10kHz to 5MHz, step 1Hz
    # amp = np.array([1e6])                     # 100Pa to 1MPa, step 1Pa
    # charge = np.array([-65])                   # -97 to 50 nC/cm²
    
    # Generate combinations using meshgrid
    print("\nGenerating parameter combinations...")
    grid = np.meshgrid(fiber_length, fiber_diameter, membrane_thickness, freq, amp, charge, indexing='ij')
    
    # Reshape to 2D array
    combinations = np.column_stack([g.ravel() for g in grid])
    
    # Convert to DataFrame
    columns = ['fiber_length', 'fiber_diameter', 'membrane_thickness', 'freq', 'amp', 'charge']
    df = pd.DataFrame(combinations, columns=columns)
    
    return df

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
    print(f"Generated {len(df)} parameter combinations")
    print(f"\nParameter ranges:")
    print(f"  • fiber_length: {df['fiber_length'].min():.1e} to {df['fiber_length'].max():.1e} m")
    print(f"  • fiber_diameter: {df['fiber_diameter'].unique()[0]:.1e} m")
    print(f"  • membrane_thickness: {df['membrane_thickness'].unique()[0]:.1e} m")
    print(f"  • frequency: {df['freq'].min():.1e} to {df['freq'].max():.1e} Hz")
    print(f"  • amplitude: {df['amp'].min():.1e} to {df['amp'].max():.1e} Pa")
    print(f"  • charge: {df['charge'].unique()[0]:.1f} nC/cm²")
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nParameter combinations saved to: {output_path}")

if __name__ == '__main__':
    main()

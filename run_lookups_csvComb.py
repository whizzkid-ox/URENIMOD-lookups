# Author: Ryo Segawa (whizznihil.kid@gmail.com)
# Run lookups.py with parameter combinations from CSV file

import os
import pandas as pd
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_lookup_with_params(script_dir, fiber_length, fiber_diameter, membrane_thickness, 
                         freq, amp, charge):
    """Execute run_lookups.py with given parameters"""
    run_lookups_path = os.path.join(script_dir, 'run_lookups.py')
    
    cmd = [
        'python',
        run_lookups_path,
        '-fiber_length', str(fiber_length),
        '-fiber_diameter', str(fiber_diameter),
        '-membrane_thickness', str(membrane_thickness),
        '-freq', str(freq),
        '-amp', str(amp),
        '-charge', str(charge)
    ]
    
    try:
        logger.info(f"Running lookup with parameters: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=script_dir)
        logger.info("Lookup completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running lookup: {e}")
        return False

def main():
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # CSV file path
    csv_path = os.path.join(script_dir, "params_csv", "hpc_lookup_parameters_20250529.csv")
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    # Read parameter combinations from CSV
    try:
        df = pd.read_csv(csv_path)
        required_columns = [
            'fiber_length', 'fiber_diameter', 'membrane_thickness',
            'freq', 'amp', 'charge'
        ]
        
        # Verify all required columns exist
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns in CSV: {missing_cols}")
            return
        
        # Run lookups for each parameter combination
        total_rows = len(df)
        successful = 0
        
        for idx, row in df.iterrows():
            logger.info(f"Processing combination {idx + 1}/{total_rows}")
            
            if run_lookup_with_params(
                script_dir=script_dir,
                fiber_length=row['fiber_length'],
                fiber_diameter=row['fiber_diameter'],
                membrane_thickness=row['membrane_thickness'],
                freq=row['freq'],
                amp=row['amp'],
                charge=row['charge']
            ):
                successful += 1
        
        logger.info(f"Completed {successful}/{total_rows} lookup combinations")
        
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")

if __name__ == "__main__":
    main()
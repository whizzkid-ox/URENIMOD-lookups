# URENIMOD Multi-Parameter Lookup Table Generation

## Overview

This directory contains modified scripts for generating URENIMOD lookup tables with multiple parameter combinations. The workflow has been enhanced to:

1. **Generate a CSV file** with all parameter combinations
2. **Process each parameter set individually** by submitting separate jobs
3. **Use variable substitution** instead of hardcoded values in the simulation command

## Modified Files

### 1. `job_multiparameters_LUT_modified.py`
- **New main script** that replaces the single-job approach
- Implements a complete workflow:
  - Step 1: Submits CSV generation job
  - Step 2: Downloads the generated CSV file
  - Step 3: Reads CSV and submits individual lookup jobs for each parameter set
- **Key improvements:**
  - Variable substitution in commands: `python run_lookups.py -fiber_length $fiber_length -fiber_diameter $fiber_diameter ...`
  - Batch processing of multiple parameter combinations
  - Individual job monitoring and management
  - Automatic job cancellation after completion

### 2. `job_parameters_generation.json`
- **Updated** to use clearer messaging about CSV generation
- Runs `lookup_parameters_csv_test.py` to generate test parameter combinations

### 3. `job_LUT_auto_download.json`
- **Used as template** by the modified Python script
- Each parameter set gets its own job configuration with specific values substituted

## Command Format Changes

### Before (Hardcoded):
```bash
python run_lookups.py -fiber_length 1e-3 -fiber_diameter 1e-6 -membrane_thickness 1.4e-9 -freq 1e6 -amp 1e6 -charge -65.0
```

### After (Variable Substitution):
```bash
python run_lookups.py -fiber_length {fiber_length} -fiber_diameter {fiber_diameter} -membrane_thickness {membrane_thickness} -freq {freq} -amp {amp} -charge {charge}
```

Where variables are replaced with actual values from each row of the CSV file.

## Usage

### Running the Multi-Parameter Workflow

```bash
cd /path/to/URENIMOD/proposal_1/scripts/lookups/GPULab
python job_multiparameters_LUT_modified.py
```

### What Happens

1. **CSV Generation** (5-10 minutes):
   - Submits `job_parameters_generation.json`
   - Generates parameter combinations CSV file
   - Downloads CSV to local system

2. **Parameter Processing** (varies by number of parameter sets):
   - Reads CSV file with parameter combinations
   - For each row in CSV:
     - Creates job configuration with specific parameter values
     - Submits lookup generation job
     - Monitors job completion
     - Cancels job after completion

3. **Results**:
   - Each parameter set generates its own lookup table files
   - Files are available for download during the job execution
   - Jobs are automatically cancelled to free GPULab resources

## Parameter Sets

The CSV generation currently uses test parameters defined in `lookup_parameters_csv_test.py`:

- **fiber_length**: [1e-4, 2e-4] (2 values)
- **fiber_diameter**: [1e-6] (1 value) 
- **membrane_thickness**: [1.4e-9] (1 value)
- **freq**: [1e6] (1 value)
- **amp**: [1e6] (1 value)
- **charge**: [-65] (1 value)

**Total combinations**: 2 parameter sets

To use different parameter ranges, modify the arrays in `lookup_parameters_csv_test.py`.

## Benefits of This Approach

1. **Scalability**: Can handle large parameter spaces by processing them in batches
2. **Resource Management**: Each job runs independently and is cancelled after completion
3. **Flexibility**: Easy to modify parameter ranges by updating the CSV generation script
4. **Monitoring**: Individual job tracking and status reporting
5. **Recovery**: If one parameter set fails, others continue processing

## Files Generated

- **CSV File**: `C:\Users\rsegawa\OneDrive - UGent\URENIMOD-data\parameter_files\lookup_parameters_YYYYMMDD.csv`
- **Lookup Tables**: Downloaded automatically during job execution to `C:\Users\rsegawa\OneDrive - UGent\URENIMOD-data\lookup_tables\unmyelinated_axon\`

## Troubleshooting

- **CSV Generation Fails**: Check the `job_parameters_generation.json` configuration and repository access
- **Individual Jobs Fail**: Monitor job logs using `gpulab-cli --cert [cert] log [job-id]`
- **Download Issues**: Verify SSH connectivity and certificate path
- **Parameter Issues**: Check that CSV file contains valid numeric values for all parameters

## Original Files

- `job_multiparameters_LUT.py`: Original single-job script (kept for reference)
- Other JSON files remain unchanged and serve their original purposes
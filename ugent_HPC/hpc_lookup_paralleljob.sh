#!/bin/bash
#PBS -l nodes=1:ppn=all
#PBS -l walltime=72:00:00

module swap env/slurm/gallade
module load worker

# Change to the base URENIMOD directory
cd /user/gent/493/vsc49347/ondemand/data/sys/myjobs/projects/URENIMOD/lookups

# Find the most recent CSV file in params_csv directory
CSV_FILE=$(ls params_csv/hpc_lookup_parameters_*.csv 2>/dev/null | sort | tail -1)

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: No parameter CSV file found in params_csv/"
    echo "Make sure to run hpc_lookup_parameters_csv.py first to generate the parameter file."
    exit 1
fi

echo "Using parameter file: $CSV_FILE"
echo "Submitting jobs using worker framework..."

# Submit the job array using worker framework
wsub -data $CSV_FILE -batch hpc_lookup_singlejob.sh
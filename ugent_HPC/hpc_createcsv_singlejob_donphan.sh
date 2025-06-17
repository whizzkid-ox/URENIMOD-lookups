#!/bin/bash
#PBS -l nodes=1:ppn=all
#PBS -l walltime=24:00:00

module load Python/3.12.3-GCCcore-13.3.0
module swap env/slurm/gallade
module load worker

cd /user/gent/493/vsc49347/ondemand/data/sys/myjobs/projects/URENIMOD/lookups

echo "Starting CSV parameter file generation..."
python3.12 hpc_lookup_parameters_csv_donphan.py

if [ $? -eq 0 ]; then
    echo "CSV parameter file generation completed successfully."
else
    echo "Error: CSV parameter file generation failed."
    exit 1
fi
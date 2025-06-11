#!/bin/bash
#PBS -l nodes=1:ppn=all
#PBS -l walltime=72:00:00

module load Python/3.12.3-GCCcore-13.3.0
# module load SciPy-bundle/2023.02-gfbf-2022b
# module load tqdm/4.64.1-GCCcore-12.2.0
# module load Tkinter/3.10.8-GCCcore-12.2.0
# module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0
# module load matplotlib/3.7.0-gfbf-2022b

cd /user/gent/493/vsc49347/ondemand/data/sys/myjobs/projects/URENIMOD/lookups

# Run the lookup script with parameters from CSV (worker framework provides these as environment variables)
python3.12 run_lookups.py -fiber_length $fiber_length -fiber_diameter $fiber_diameter -membrane_thickness $membrane_thickness -freq $freq -amp $amp -charge $charge
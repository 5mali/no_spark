#!/bin/bash

#SBATCH --job-name=batch_run
#SBATCH --output=batch_run.out
#SBATCH -p p
#SBATCH -N 1
#SBATCH --cpus-per-task 30
source activate eno
srun echo "Number of cores: "
srun parallel --number-of-cores
srun echo "Number of CPUS: "
srun parallel --number-of-cpus
srun echo "Doing jobs: "
echo ${@}
echo "***************"
srun parallel ./run_python_script.sh ::: ${@}
source deactivate

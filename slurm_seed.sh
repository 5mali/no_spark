#!/bin/bash
#
######SBATCH --job-name=$(1)
######SBATCH --output=${1}job.txt
#SBATCH -p knm
###### #SBATCH --gres=gpu:1
#SBATCH -n 15
######SBATCH --time=60:00
###### #SBATCH --mem-per-cpu=100

#srun hostname
#srun sleep 60
#srun conda --version
#srun python --version
#srun conda env list
#srun pwd
#srun cd ~/repos/no_spark
source activate eno
echo $1
srun ~/repos/no_spark/parallel_seed_test.sh ${1} 

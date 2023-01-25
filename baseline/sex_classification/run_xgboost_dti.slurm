#!/bin/bash
#SBATCH --job-name=Xgboost-DTI-sex         # Job name
#SBATCH --nodes=1                      # Total # of nodes
#SBATCH --nodelist=node1
#SBATCH --ntasks-per-node=1           # # of tasks per node
#SBATCH --cpus-per-task=16              # # of cores per task
#SBATCH --time=24:00:00                # Time limit (hh:mm:ss)
#SBATCH --output=../../slurm_logs/%A_%x.out         # Standard error
#SBATCH --error=../../slurm_logs/%A_%x.err          # Standard output
#SBATCH --mail-user=stellasybae@snu.ac.kr  # Email address
#SBATCH --mail-type=ALL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --gpus-per-task=1

#export MASTER_ADDR=$(hostname)

source /usr/anaconda3/etc/profile.d/conda.sh
conda activate stella_mm

python Xgboost_classification_DTI.py
#!/bin/bash
#
#SBATCH --partition=itp
#SBATCH --nodes=1-1
#SBATCH --ntasks=1
#SBATCH --mail-user=fschubert@itp.uni-frankfurt.de
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#
#SBATCH --job-name=mc_sweep
#SBATCH --mem=8000
#SBATCH --cpus-per-task=12
#SBATCH --time=10-00:00:00

# does something...
export OMP_NUM_THREADS=$SLURM_NTASKS_PER_NODE

#RUNPATH=/home/fschubert/work/repos/DrivenRandNetworkCriticality/code/
#cd $RUNPATH
source /home/fschubert/work/py36/bin/activate
python3 var_input_var_target_sweep_XOR.py

#!/bin/bash
#
#SBATCH --job-name=network-sweep
#SBATCH --output=res.txt
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=15000
#SBATCH --partition=itp

srun var_input_var_target_sweep.py --sigmaw 2.0 --filename sim_results_2_0.npz

#! /bin/bash

#SBATCH -p itp
#SBATCH -n 1
#SBATCH -c 1


source ~/work/py36/bin/activate

cd ~/work/repos/DrivenRandNetworkCriticality/code

python3 -m ESN_code.simulation.var_predict_scaling homogeneous_independent_gaussian

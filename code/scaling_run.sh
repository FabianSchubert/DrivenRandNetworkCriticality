#!/usr/bin/env bash
for k in $(seq 5)
do
	for N in 100 200 300 400 500 1000
	do
		python -m ESN_code.simulation.param_sweep $1 --N $N --n_sweep_sigm_e 1 --n_sweep_sigm_t 1
	done
done

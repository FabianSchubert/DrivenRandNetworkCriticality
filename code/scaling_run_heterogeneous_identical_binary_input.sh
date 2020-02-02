#!/usr/bin/env bash
for k in $(seq 10)
do
	for N in 100 200 300 400 500 1000
	do
		python -m heterogeneous_identical_binary_input_ESN.simulation.param_sweep $N 1 30
	done
done

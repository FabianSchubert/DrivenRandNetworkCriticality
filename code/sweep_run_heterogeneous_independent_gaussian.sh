#!/usr/bin/env bash
for k in $(seq 10)
do
	python -m heterogeneous_independent_gaussian_input_ESN.simulation.param_sweep
done

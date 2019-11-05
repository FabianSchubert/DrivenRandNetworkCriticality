#!/usr/bin/env bash
for k in $(seq 10)
do
	python -m homogeneous_independent_gaussian_input_ESN.simulation.param_sweep 500
done

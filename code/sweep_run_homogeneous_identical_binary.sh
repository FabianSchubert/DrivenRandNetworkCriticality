#!/usr/bin/env bash
for k in $(seq 10)
do
	python -m homogeneous_identical_binary_input_ESN.simulation.param_sweep
done

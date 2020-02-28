#!/usr/bin/env bash

for i in $(seq 4)
do
	python3 -m ESN_code.simulation.param_sweep homogeneous_independent_gaussian
	python3 -m ESN_code.simulation.param_sweep homogeneous_identical_binary
	python3 -m ESN_code.simulation.param_sweep heterogeneous_independent_gaussian
	python3 -m ESN_code.simulation.param_sweep heterogeneous_identical_binary
done

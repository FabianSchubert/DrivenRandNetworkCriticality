#!/usr/bin/env bash

for k in $(seq 5)
do
    echo $k of 5
    python3 -m ESN_code.simulation.param_sweep_performance heterogeneous_identical_binary
    python3 -m ESN_code.simulation.param_sweep_performance heterogeneous_independent_gaussian
    python3 -m ESN_code.simulation.param_sweep_performance homogeneous_identical_binary
    python3 -m ESN_code.simulation.param_sweep_performance homogeneous_independent_gaussian
done

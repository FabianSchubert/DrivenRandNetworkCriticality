#!/usr/bin/env bash
for k in $(seq 10)
do
	python -m ESN_code.simulation.param_sweep $1
done

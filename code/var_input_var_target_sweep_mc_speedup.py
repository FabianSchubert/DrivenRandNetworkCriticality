#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from simulation import driven_net
from echo_state_tests import test_memory_cap, test_echo_state_prop
from tqdm import tqdm
import os
import sys
import argparse

from scipy.integrate import quad
from scipy.integrate import romberg
from scipy.integrate import quadrature
from scipy.optimize import root


def sigm_int_func(x, g, sigm_ext, sigm_targ, sigm_w):
    return gauss(x, 0., (sigm_ext**2 + sigm_targ**2*sigm_w**2)**.5) * tanh_func(g * x)**2


def f_gain_root(x, sigm_ext, sigm_targ, sigm_w):
    int, err = quad(sigm_int_func, -np.infty, np.infty,
                    args=(x, sigm_ext, sigm_targ, sigm_w), epsrel=0.000001)
    #int = romberg(sigm_int_func,-10.,10.,args=(x,sigm_ext,sigm_targ))
    #int, err = quadrature(sigm_int_func,-10.,10.,args=(x,sigm_ext,sigm_targ))
    return int - sigm_targ**2


def find_consist_gain(sigm_ext, sigm_targ, sigm_w):
    sol = root(f_gain_root, (1. + sigm_ext**2 / sigm_targ**2) **
               (-.5), (sigm_ext, sigm_targ, sigm_w))
    return sol['x']


def gauss(x, m, s):
    return np.exp(-(x - m)**2 / (2. * s**2)) / (2. * np.pi * s**2)**.5


def tanh_func(x):
    # return x
    # return x - x**3/3.
    return np.tanh(x)

# input generator for testing memory capacity
def gen_input(t):
    return np.random.normal()


from standard_params import *

n_sweep_std_in = 30
n_sweep_std_act_target = 30

std_in_sweep_range = np.linspace(0.001,1.5,n_sweep_std_in)
std_act_target_sweep_range = np.linspace(0.001,.9,n_sweep_std_act_target)

g_pred = np.ndarray((n_sweep_std_in, n_sweep_std_act_target))
mc = np.ndarray((n_sweep_std_in, n_sweep_std_act_target))

sigm_w = 1.

for k in range(n_sweep_std_in):
    for l in range(n_sweep_std_act_target):
        g_pred[k, l] = find_consist_gain(
            std_in_sweep_range[k], std_act_target_sweep_range[l], sigm_w)

        W = np.random.normal(0.,sigm_w/(N_net_def*cf_net_def)**.5,(N_net_def))
        gen_temp = lambda t: gen_input(t)*std_in_sweep_range[k]
        MC, MC_sum = test_memory_cap((DN.W.T*DN.gain).T,50,5000,gen_temp,0.1)


plt.pcolormesh(g_pred)
plt.colorbar()
plt.show()

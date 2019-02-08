#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import pdb

from scipy.integrate import quad
from scipy.optimize import root


def gauss(x, m, s):
    return np.exp(-(x - m)**2 / (2. * s**2)) / (2. * np.pi * s**2)**.5


def tanh_func(x):
    # return x
    # return x - x**3/3.
    return np.tanh(x)


def int_func(x, g, sigm_ext, sigm_targ):
    return gauss(x, 0., (sigm_ext**2 + sigm_targ**2)**.5) * tanh_func(g * x)**2


def f(x, sigm_ext, sigm_targ):
    int, err = quad(int_func, -2., 2., args=(x, sigm_ext, sigm_targ))

    return int - sigm_targ**2


def find_consist_gain(sigm_ext, sigm_targ):
    sol = root(f, 1., (sigm_ext, sigm_targ))
    return sol['x']

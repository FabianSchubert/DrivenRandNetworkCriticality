#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from simulation import driven_net
from tqdm import tqdm
### Parameters
### Import standard parameters
from standard_params import *

folder = "../data/"

file = "gain_conv_sweep.npz"

std_in = 0.5
std_targ = 0.5
mu_targ = 0.

n_t_def = 100000

N_net_range = [100,200,300,400,500,1000,1500,2000,2500,3000,4000,5000,6000,7000,8000,9000,10000]
#N_net_range = [10000]

n_sweep = len(N_net_range)

gain_pop_std = np.ndarray((n_sweep))


for k in tqdm(range(n_sweep)):

    DN = driven_net(N_net_range[k],
                cf_net_def,
                std_conn_def,
                std_in,
                mu_targ,
                std_targ,
                mu_bias_def,
                mu_gain_def,
                mu_trail_av_error_def,
                n_t_def,
                t_ext_off_def,
                x_rec = False,
                I_in_rec = False,
                gain_rec = False,
                var_mean_rec = False)

    DN.run_sim()

    gain_pop_std[k] = DN.gain.std()

parameters = {
    "cf":cf_net_def,
    "std_conn":std_conn_def,
    "mu_act_target":mu_targ,
    "mu_gain":mu_gain_def,
    "mu_bias":mu_bias_def,
    "mu_trail_av_error":mu_trail_av_error_def,
    "t_ext_off":t_ext_off_def,
    "n_t":n_t_def
}


np.savez_compressed(folder + file,
    N_net_range=N_net_range,
    gain_pop_std=gain_pop_std,
    parameters=parameters
    )

plt.plot(N_net_range,gain_pop_std)

import pdb; pdb.set_trace()

#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from simulation import driven_net
from tqdm import tqdm
import os

### Parameters
N_net_def = 1000

std_conn_def = 1.

cf_net_def = 0.1

mu_gain_def = 0.001

n_t_def = 50000

t_ext_off_def = 50000

n_sweep_std_in = 20
n_sweep_std_act_target = 20

std_in_sweep_range = np.linspace(0.,1.,n_sweep_std_in)
std_act_target_sweep_range = np.linspace(0.,0.25,n_sweep_std_act_target)
max_l_list = np.ndarray((n_sweep_std_in,n_sweep_std_act_target))
gain_list = np.ndarray((n_sweep_std_in,n_sweep_std_act_target,N_net_def))


for k in tqdm(range(n_sweep_std_in)):
    for l in tqdm(range(n_sweep_std_act_target)):
        DN = driven_net(N_net_def,
                    cf_net_def,
                    std_conn_def,
                    std_in_sweep_range[k],
                    std_act_target_sweep_range[l],
                    mu_gain_def,
                    n_t_def,
                    t_ext_off_def,
                    x_rec = False,
                    I_in_rec = False,
                    gain_rec = False,
                    var_mean_rec = False)
        DN.run_sim()
        max_l_list[k,l] = np.abs(np.linalg.eigvals((DN.W.T*DN.gain).T)).max()
        gain_list[k,l,:] = DN.gain


if not os.path.exists("../data/max_lyap_sweep/"):
    os.makedirs("../data/max_lyap_sweep/")
np.savez_compressed("../data/max_lyap_sweep/sim_results.npz",
max_l_list = max_l_list,
gain_list = gain_list,
std_in_sweep_range = std_in_sweep_range,
std_act_target_sweep_range = std_act_target_sweep_range)

'''
### Plotting

t_ax = np.array(range(n_t_def-100,n_t_def))

textwidth = 5.5532
std_figsize = (textwidth/2.,2.)
dpi_screen = 120


#============================
fig_max_l_sweep, ax_max_l_sweep = plt.subplots(1,1,figsize=std_figsize, dpi=dpi_screen)
pcm = ax_max_l_sweep.pcolormesh(std_act_target_sweep_range,std_in_sweep_range,np.log(max_l_list))
plt.colorbar(mappable=pcm,ax=ax_max_l_sweep)
ax_max_l_sweep.contour(std_act_target_sweep_range,std_in_sweep_range,np.log(max_l_list), [0.],colors='r')

ax_max_l_sweep.set_xlabel("$\\sigma$ act. target")
ax_max_l_sweep.set_ylabel("$\\sigma$ ext. input")

fig_max_l_sweep.tight_layout()
fig_max_l_sweep.savefig("../plots/max_l_sweep.png", dpi=300)

plt.show()
#############################
'''

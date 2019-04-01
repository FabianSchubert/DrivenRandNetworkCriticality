#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from simulation import driven_net
from echo_state_tests import test_memory_cap, test_echo_state_prop
from tqdm import tqdm
import os
import sys
import argparse

# input generator for testing memory capacity
def gen_input(t):
    return np.random.normal()

path = "/mnt/ceph/fschubert/data/max_lyap_sweep/"
#filename = "sim_results.npz"
filename = "sim_results_test.npz"


### Parameters
### Import standard parameters
from standard_params import *

n_t_def = 100000

n_sweep_std_in = 30
n_sweep_std_act_target = 30

det_max_l = False
det_mc = True
det_esp = True

std_in_sweep_range = np.linspace(0.,1.5,n_sweep_std_in)
std_act_target_sweep_range = np.linspace(0.,.9,n_sweep_std_act_target)

if det_max_l:
    max_l_list = np.ndarray((n_sweep_std_in,n_sweep_std_act_target))
else:
    max_l_list = None

gain_list = np.ndarray((n_sweep_std_in,n_sweep_std_act_target,N_net_def))

mean_var_list = np.ndarray((n_sweep_std_in,n_sweep_std_act_target))

trail_av_hom_error_list = np.ndarray((n_sweep_std_in,n_sweep_std_act_target))

if det_mc:
    mem_cap_list = np.ndarray((n_sweep_std_in,n_sweep_std_act_target))
else:
    mem_cap_list = None

if det_esp:
    echo_state_prop_list = np.ndarray((n_sweep_std_in,n_sweep_std_act_target))
else:
    echo_state_prop_list = None

W_list = np.ndarray((n_sweep_std_in,n_sweep_std_act_target,N_net_def,N_net_def))

params_list = [[None for l in range(n_sweep_std_act_target)] for k in range(n_sweep_std_in)]


ind_std_in_sample_data = None
ind_std_act_target_sample_data = None

parser = argparse.ArgumentParser(description='''Run a parameter sweep over sigma_ext
and sigma_target for a driven random recurrent network.''')

parser.add_argument('--sigmaw', type=float, help='specify the standard deviation of neural weights.')
parser.add_argument('--filename', help='specify filename.')

args = parser.parse_args()

if args.sigmaw!=None:
    std_conn_def = args.sigmaw
if args.filename!=None:
    filename=args.filename




for k in tqdm(range(n_sweep_std_in)):
    for l in tqdm(range(n_sweep_std_act_target)):
        if k==ind_std_in_sample_data and l==ind_std_act_target_sample_data:
            DN = driven_net(N_net_def,
                        cf_net_def,
                        std_conn_def,
                        std_in_sweep_range[k],
                        mu_act_target_def,
                        std_act_target_sweep_range[l],
                        eps_bias_def,
                        eps_gain_def,
                        eps_trail_av_error_def,
                        eps_trail_av_act_def,
                        n_t_def,
                        t_ext_off_def,
                        x_rec = False,
                        x_trail_av_rec = False,
                        bias_rec = False,
                        I_in_rec = False,
                        gain_rec = False,
                        var_mean_rec = False)

        else:
            DN = driven_net(N_net_def,
                        cf_net_def,
                        std_conn_def,
                        std_in_sweep_range[k],
                        mu_act_target_def,
                        std_act_target_sweep_range[l],
                        eps_bias_def,
                        eps_gain_def,
                        eps_trail_av_error_def,
                        eps_trail_av_act_def,
                        n_t_def,
                        t_ext_off_def,
                        x_rec = False,
                        x_trail_av_rec = False,
                        bias_rec = False,
                        I_in_rec = False,
                        gain_rec = False,
                        var_mean_rec = False)

        DN.run_sim()

        if det_max_l:
            max_l_list[k,l] = np.abs(np.linalg.eigvals((DN.W.T*DN.gain).T)).max()
        gain_list[k,l,:] = DN.gain
        W_list[k,l,:,:] = DN.W
        mean_var_list[k,l] = (DN.x_net**2).mean()
        trail_av_hom_error_list[k,l] = DN.trail_av_hom_error

        if det_mc:
            gen_temp = lambda t: gen_input(t)*std_in_sweep_range[k]
            MC, MC_sum = test_memory_cap((DN.W.T*DN.gain).T,50,1000,gen_temp,0.1)
            mem_cap_list[k,l] = MC_sum

        if det_esp:
            gen_temp = lambda t: gen_input(t)*std_in_sweep_range[k]
            esp_test, esp_test_rec = test_echo_state_prop((DN.W.T*DN.gain).T,10000,0.001,10.**-10,gen_temp,x_init=DN.x_net)
            echo_state_prop_list[k,l] = esp_test

        params_list[k][l] = DN.get_params()

        '''
        if k==ind_std_in_sample_data and l==ind_std_act_target_sample_data:
            if not os.path.exists("../data/max_lyap_sweep/"):
                os.makedirs("../data/max_lyap_sweep/")
            np.savez_compressed("../data/max_lyap_sweep/sim_results_sample_run_pretest.npz",
            std_in = std_in_sweep_range[ind_std_in_sample_data],
            std_act_target = std_act_target_sweep_range[ind_std_act_target_sample_data],
            x_rec = DN.x_net_rec,
            gain_rec = DN.gain_rec,
            var_mean_rec = DN.var_mean_rec)
        '''

if not os.path.exists(path):
    os.makedirs(path)
np.savez_compressed(path+filename,
max_l_list = max_l_list,
gain_list = gain_list,
W_list = W_list,
trail_av_hom_error_list = trail_av_hom_error_list,
mem_cap_list = mem_cap_list,
echo_state_prop_list = echo_state_prop_list,
std_in_sweep_range = std_in_sweep_range,
std_act_target_sweep_range = std_act_target_sweep_range,
params_list = params_list)

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

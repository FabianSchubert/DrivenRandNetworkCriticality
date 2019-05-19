#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from simulation import driven_net
from echo_state_tests import test_memory_cap, test_echo_state_prop, test_XOR
from tqdm import tqdm
import os
import sys
import argparse

import time

# input generator for testing memory capacity
#def gen_input(t):
#   return np.random.normal()

path = "/mnt/ceph/fschubert/data/max_lyap_sweep/"
#filename = "sim_results.npz"
filename = "sim_results_test.npz"


### Parameters
### Import standard parameters
from standard_params import *

n_t_def = 100000

mu_act_target_def = (np.random.rand(N_net_def)-.5)*2.*0.1

n_sweep_std_in = 30
n_sweep_std_act_target = 30

det_max_l = True
det_mc = True
det_esp = True

det_mc_xor = True

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

if det_mc_xor:
    mc_xor_list = np.ndarray((n_sweep_std_in,n_sweep_std_act_target))
else:
    mc_xor_list = None

#W_list = np.ndarray((n_sweep_std_in,n_sweep_std_act_target,N_net_def,N_net_def))

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


# Init Weights
W = np.random.normal(0., std_conn_def, (N_net_def, N_net_def))
W *= (np.random.rand(N_net_def, N_net_def) <=
           cf_net_def) / (N_net_def * cf_net_def)**.5
W[range(N_net_def), range(N_net_def)] = 0.

t0 = time.time()

f = open("sim_text_output.txt","a")

for k in range(n_sweep_std_in):#tqdm(range(n_sweep_std_in)):
    for l in range(n_sweep_std_act_target):#tqdm(range(n_sweep_std_act_target)):

        t_el = (time.time()-t0)/60.
        n_el = l+k*n_sweep_std_act_target
        if k==0 and l==0:
            print(str(n_el)+"/"+str(n_sweep_std_act_target*n_sweep_std_in) + " " + '{:.2f}'.format(t_el) +  " minutes elapsed")
            f.write(str(n_el)+"/"+str(n_sweep_std_act_target*n_sweep_std_in) + " " + '{:.2f}'.format(t_el) +  " minutes elapsed\n")
        else:
            t_rest = (n_sweep_std_act_target*n_sweep_std_in - n_el)*t_el/n_el

            if t_rest >= 60.:
                str_t_rest = str(int(t_rest/60.)) +  ":" + str(int(t_rest%60.)) + " h"
            else:
                str_t_rest = '{:.2f}'.format(t_rest) + " min"
            print(str(l+k*n_sweep_std_act_target)+"/"+str(n_sweep_std_act_target*n_sweep_std_in) + " " + '{:.2f}'.format((time.time()-t0)/60.) +  " minutes elapsed, approx. " + str_t_rest + " to go")
            f.write(str(l+k*n_sweep_std_act_target)+"/"+str(n_sweep_std_act_target*n_sweep_std_in) + " " + '{:.2f}'.format((time.time()-t0)/60.) +  " minutes elapsed, approx. " + str_t_rest + " to go\n")
        f.flush()
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

        DN.W = W
        DN.run_sim()

        if det_max_l:
            max_l_list[k,l] = np.abs(np.linalg.eigvals((DN.W.T*DN.gain).T)).max()
        gain_list[k,l,:] = DN.gain
        #W_list[k,l,:,:] = DN.W
        mean_var_list[k,l] = (DN.x_net**2).mean()
        trail_av_hom_error_list[k,l] = DN.trail_av_hom_error

        #(W,t_back_max,n_learn_samples,break_low_threshold,tresh_av_wind,input_gen,reg_fact)

        if det_mc:
            gen_temp = lambda t: np.random.normal()*std_in_sweep_range[k]
            #test_memory_cap(W,t_back_max,n_learn_samples,break_low_threshold,tresh_av_wind,input_gen,reg_fact)
            MC, MC_sum = test_memory_cap(W,DN.gain,DN.bias,120,1000,0.01,10,gen_temp,0.1)
            mem_cap_list[k,l] = MC_sum

        if det_esp:
            #gen_temp = lambda t: np.random.normal(0.,1.,(N_net_def))*std_in_sweep_range[k]
            gen_tenp = lambda t: ((np.random.rand()<=0.5)*2.*std_in_sweep_range[k])
            esp_test, esp_test_rec = test_echo_state_prop(W,DN.gain,DN.bias,10000,0.001,10.**-10,gen_temp,x_init=DN.x_net)
            echo_state_prop_list[k,l] = esp_test

        if det_mc_xor:
            MC_XOR, MC_XOR_sum = test_XOR(W,DN.gain,DN.bias,25,1000,0.01,10,std_in_sweep_range[k],0.1)
            mc_xor_list[k,l] = MC_XOR_sum

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
W = W,
trail_av_hom_error_list = trail_av_hom_error_list,
mem_cap_list = mem_cap_list,
mem_cap_xor_list = mc_xor_list,
echo_state_prop_list = echo_state_prop_list,
std_in_sweep_range = std_in_sweep_range,
std_act_target_sweep_range = std_act_target_sweep_range,
params_list = params_list)

f.write("done\n")

f.close()

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

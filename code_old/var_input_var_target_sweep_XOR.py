import numpy as np

from echo_state_tests import test_memory_cap, test_echo_state_prop, test_XOR, gen_in_out_one_in_subs
from tqdm import tqdm
import os
import sys
import argparse

from multiprocessing.dummy import Pool
from multiprocessing import Lock

from esn_module import esn

import time

import pdb

# input generator for testing memory capacity
#def gen_input(t):
#   return np.random.normal()

path = "/mnt/ceph/fschubert/data/max_lyap_sweep/"
#path = "/home/fabian/work/"
#filename = "sim_results.npz"
filename = "xor_lyap_esp_2.npz"


### Parameters
### Import standard parameters
from standard_params import *

n_t_def = 150000

mu_act_target_def = (np.random.rand(N_net_def)-.5)*2.*0.1

n_sweep_std_in = 30
n_sweep_std_act_target = 30

det_max_l = True
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



#ESN = esn(N=N_net_def,cf=cf_net_def)

n_threads = 4

pool = Pool(n_threads)

parallel_it = []

lock_parall = Lock()

for k in range(n_sweep_std_in):#tqdm(range(n_sweep_std_in)):
    for l in range(n_sweep_std_act_target):#tqdm(range(n_sweep_std_act_target)):
        parallel_it.append([lock_parall,k,l])



t0 = time.time()
n_el = 0



def run_simulation(parameters):

    lock,k,l = parameters

    global n_el,f

    lock.acquire()
    n_el += 1

    t_el = (time.time()-t0)/60.

    t_rest = (n_sweep_std_act_target*n_sweep_std_in - n_el)*t_el/n_el

    if t_rest >= 60.:
        str_t_rest = str(int(t_rest/60.)) +  ":" + str(int(t_rest%60.)) + " h"
    else:
        str_t_rest = '{:.2f}'.format(t_rest) + " min"
    print(str(l+k*n_sweep_std_act_target)+"/"+str(n_sweep_std_act_target*n_sweep_std_in) + " " + '{:.2f}'.format((time.time()-t0)/60.) +  " minutes elapsed, approx. " + str_t_rest + " to go")
    f.write(str(l+k*n_sweep_std_act_target)+"/"+str(n_sweep_std_act_target*n_sweep_std_in) + " " + '{:.2f}'.format((time.time()-t0)/60.) +  " minutes elapsed, approx. " + str_t_rest + " to go\n")
    f.flush()
    lock.release()

    #k,l = indices

    ESN = esn(sigm_act_target=std_act_target_sweep_range[l],mu_act_target=np.random.normal(0.,0.01,(1000)))

    u_in = (np.random.rand(n_t_def) < .5)*1.*2.*std_in_sweep_range[k]

    ESN.run_hom_adapt(u_in,show_progress=False)

    max_l = np.abs(np.linalg.eigvals(ESN.W_gain())).max()

    gain = ESN.gain

    #(W,t_back_max,n_learn_samples,break_low_threshold,tresh_av_wind,input_gen,reg_fact)

    u_in = (np.random.rand(5000) < .5)*1.*2.*std_in_sweep_range[k]

    dist,p,esp_test = ESN.test_ESP(u_in,10.**-10.,show_progress=False)
    echo_state_prop = esp_test


    MC_XOR, MC_XOR_sum = test_XOR(ESN,20,5000,0.01,10,std_in_sweep_range[k],show_progress=False)
    mc_xor = MC_XOR_sum

    params = ESN.get_params()



    return [max_l,gain,echo_state_prop,mc_xor,params]

f = open("sim_text_output_2.txt","a")

results = pool.map(run_simulation,parallel_it)

pool.close()
pool.join()

f.close()

for i,p in enumerate(parallel_it):

    k = p[1]
    l = p[2]

    max_l_list[k,l] = results[i][0]

    gain_list[k,l,:] = results[i][1]

    echo_state_prop_list[k,l] = results[i][2]

    mc_xor_list[k,l] = results[i][3]

    params_list[k][l] = results[i][4]


'''
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

        ESN.sigm_act_target=std_act_target_sweep_range[l]

        u_in = (np.random.rand(n_t_def) < .5)*1.*2.*std_in_sweep_range[k]

        ESN.run_hom_adapt(u_in)

        if det_max_l:
            max_l_list[k,l] = np.abs(np.linalg.eigvals(ESN.W_gain())).max()

        gain_list[k,l,:] = ESN.gain

        #(W,t_back_max,n_learn_samples,break_low_threshold,tresh_av_wind,input_gen,reg_fact)

        if det_esp:
            u_in = (np.random.rand(10000) < .5)*1.*2.*std_in_sweep_range[k]
            dist,p,esp_test = ESN.test_ESP(u_in,10.**-10.,show_progress=True)
            echo_state_prop_list[k,l] = esp_test

        if det_mc_xor:
            MC_XOR, MC_XOR_sum = test_XOR(ESN,20,5000,0.01,10,std_in_sweep_range[k],show_progress=True)
            mc_xor_list[k,l] = MC_XOR_sum

        params_list[k][l] = ESN.get_params()

'''


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
mem_cap_xor_list = mc_xor_list,
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

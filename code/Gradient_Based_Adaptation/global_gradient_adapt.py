#!/usr/bin/env python3

import numpy as np

from tqdm import tqdm

from sim_modules.rnn import RNN

from sim_modules.testfuncs import gen_in_out_one_in_subs

data_path = "../../data/Gradient_Based_Adaptation/"

n_samples = 10

n_sweep = 50

gain_arr = np.linspace(0.2,1.5,n_sweep)
tau_arr = np.array([1,5,10,15])

T_sim = 10000

delta_a_global_mean_RTRL_global = np.ndarray((n_samples,tau_arr.shape[0],n_sweep))
delta_a_global_mean_RTRL_local = np.ndarray((n_samples,tau_arr.shape[0],n_sweep))



for s in tqdm(range(n_samples)):

    for n in tqdm(range(tau_arr.shape[0])):

        for k in tqdm(range(n_sweep)):

            ### Global Gradient
            rnn = RNN()
            rnn.a *= gain_arr[k]
            rnn.eps_a = 0.

            u_in,u_out = gen_in_out_one_in_subs(T_sim,tau_arr[n])

            #'''
            t,y,X,X_r,X_e,a,delta_a,w_out,E,W = rnn.learn_gain(u_in,u_out,mode='global_grad_global_gain',
                                                               tau_batch_w_out=1.,T_skip_rec=1,
                                                               show_progress=False,return_dyda=False)

            delta_a_global_mean_RTRL_global[s,n,k] = delta_a[rnn.N*4:].mean(axis=0)
            ###


            ### Local Gradient
            rnn = RNN()
            rnn.a *= gain_arr[k]
            rnn.eps_a = 0.

            u_in,u_out = gen_in_out_one_in_subs(T_sim,tau_arr[n])

            #'''
            t,y,X,X_r,X_e,a,delta_a,w_out,E,W = rnn.learn_gain(u_in,u_out,mode='local_grad_global_gain',
                                                               tau_batch_w_out=1.,T_skip_rec=1,
                                                               show_progress=False,return_dyda=False)
            delta_a_global_mean_RTRL_local[s,n,k] = delta_a[rnn.N*4:].mean(axis=0)
            ###

np.save(data_path + "delta_a_global_mean_RTRL_global.npy",delta_a_global_mean_RTRL_global)
np.save(data_path + "delta_a_global_mean_RTRL_local.npy",delta_a_global_mean_RTRL_local)

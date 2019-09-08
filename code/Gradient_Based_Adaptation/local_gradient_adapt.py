#!/usr/bin/env python3

import numpy as np

from tqdm import tqdm

from sim_modules.rnn import RNN

from sim_modules.testfuncs import gen_in_out_one_in_subs

data_path = "../../data/Gradient_Based_Adaptation/testing/"

N=50

n_samples = 1

n_sweep = 10

gain_arr = np.linspace(0.2,1.5,n_sweep)
tau_arr = np.array([1,5,10,15])
#tau_arr = np.array([15])

T_batch_w_out = N*2

T_sim = 50000 + T_batch_w_out

T_skip_rec = 10

T_batch_w_out_skip = int(T_batch_w_out/T_skip_rec)

delta_a_local_mean_RTRL_global = np.ndarray((n_samples,tau_arr.shape[0],n_sweep))
delta_a_local_mean_RTRL_local = np.ndarray((n_samples,tau_arr.shape[0],n_sweep))

delta_a_local_err_RTRL_global = np.ndarray((n_samples,tau_arr.shape[0],n_sweep))
delta_a_local_err_RTRL_local = np.ndarray((n_samples,tau_arr.shape[0],n_sweep))

delta_a_local_std_RTRL_global = np.ndarray((n_samples,tau_arr.shape[0],n_sweep))
delta_a_local_std_RTRL_local = np.ndarray((n_samples,tau_arr.shape[0],n_sweep))

import matplotlib.pyplot as plt
plt.ion()
import pdb



for s in tqdm(range(n_samples)):

    for n in tqdm(range(tau_arr.shape[0])):

        ### Local Gradient
        rnn = RNN(N=N)

        rnn.eps_a = 0.



        for k in tqdm(range(n_sweep)):


            #'''
            ### Global Gradient

            rnn.a[:] = gain_arr[k]

            rho = (rnn.a**2.).mean()**.5

            u_in, u_out = gen_in_out_one_in_subs(T_sim,tau_arr[n])

            #'''
            t,delta_a,w_out,W = rnn.learn_gain(u_in,u_out,mode='global_grad_local_gain',
                                                               tau_batch_w_out=1.,
                                                               show_progress=True,return_dyda=False,
                                                               return_y=False,
                                                               return_X=False,return_X_r=False,
                                                               return_X_e=False,return_a=False,
                                                               return_Err=False,
                                                               T_stop_learn_wout=T_sim,
                                                               T_batch_w_out=T_batch_w_out,
                                                               T_skip_rec=T_skip_rec)



            delta_a_local_mean_RTRL_global[s,n,k] = (rnn.a*delta_a[T_batch_w_out_skip:,:]).sum(axis=1).mean()/(rho*rnn.N)
            delta_a_local_err_RTRL_global[s,n,k] = ((rnn.a*delta_a[T_batch_w_out_skip:,:]).sum(axis=1).std()/(rho*rnn.N))/(T_sim - rnn.N*2)**.5
            delta_a_local_std_RTRL_global[s,n,k] = ((rnn.a*delta_a[T_batch_w_out_skip:,:]).sum(axis=1).std()/(rho*rnn.N))
            ###
            #'''

            ### Local Gradient
            rnn.a[:] = gain_arr[k]

            rho = (rnn.a**2.).mean()**.5

            u_in, u_out = gen_in_out_one_in_subs(T_sim,tau_arr[n])

            #'''
            t,delta_a,w_out,W = rnn.learn_gain(u_in,u_out,mode='local_grad_local_gain',
                                                               tau_batch_w_out=1.,
                                                               show_progress=True,return_dyda=False,
                                                               return_y=False,
                                                               return_X=False,return_X_r=False,
                                                               return_X_e=False,return_a=False,
                                                               return_Err=False,
                                                               T_stop_learn_wout=T_sim,
                                                               T_batch_w_out=T_batch_w_out,
                                                               T_skip_rec=T_skip_rec)



            delta_a_local_mean_RTRL_local[s,n,k] = (rnn.a*delta_a[T_batch_w_out_skip:,:]).sum(axis=1).mean()/(rho*rnn.N)
            delta_a_local_err_RTRL_local[s,n,k] = ((rnn.a*delta_a[T_batch_w_out_skip:,:]).sum(axis=1).std()/(rho*rnn.N))/(T_sim - rnn.N*2)**.5
            delta_a_local_std_RTRL_local[s,n,k] = ((rnn.a*delta_a[T_batch_w_out_skip:,:]).sum(axis=1).std()/(rho*rnn.N))
            #delta_a_RTRL_local = delta_a[rnn.N*4:,:]
            #pdb.set_trace()

            #delta_a_local_mean_RTRL_local[s,n,k] = (((rnn.a+delta_a[rnn.N*4:,:])**2.).mean(axis=1)**.5-(rnn.a**2.).mean()**.5).mean()
            ###

        plt.plot(gain_arr,delta_a_local_mean_RTRL_local[s,n,:])
        #plt.fill_between(gain_arr,delta_a_local_mean_RTRL_local[0,0,:]-delta_a_local_std_RTRL_local[0,0,:],delta_a_local_mean_RTRL_local[0,0,:]+delta_a_local_std_RTRL_local[0,0,:],alpha=.5)
        plt.fill_between(gain_arr,delta_a_local_mean_RTRL_local[s,n,:]-delta_a_local_err_RTRL_local[s,n,:],delta_a_local_mean_RTRL_local[s,n,:]+delta_a_local_err_RTRL_local[s,n,:],alpha=.5)
        plt.show()
        pdb.set_trace()

    np.save(data_path + "delta_a_local_mean_RTRL_global_sample_" + str(s) + ".npy",delta_a_local_mean_RTRL_global[s,:,:])
    np.save(data_path + "delta_a_local_mean_RTRL_local_sample_" + str(s) + ".npy",delta_a_local_mean_RTRL_local[s,:,:])

    np.save(data_path + "delta_a_local_err_RTRL_global_sample_" + str(s) + ".npy",delta_a_local_err_RTRL_global[s,:,:])
    np.save(data_path + "delta_a_local_err_RTRL_local_sample_" + str(s) + ".npy",delta_a_local_err_RTRL_local[s,:,:])

np.save(data_path + "delta_a_local_mean_RTRL_global.npy",delta_a_local_mean_RTRL_global)
np.save(data_path + "delta_a_local_mean_RTRL_local.npy",delta_a_local_mean_RTRL_local)

np.save(data_path + "delta_a_local_err_RTRL_global.npy",delta_a_local_err_RTRL_global)
np.save(data_path + "delta_a_local_err_RTRL_local.npy",delta_a_local_err_RTRL_local)

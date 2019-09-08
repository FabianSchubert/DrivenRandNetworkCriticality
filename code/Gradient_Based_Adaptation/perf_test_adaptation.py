#!/usr/bin/env python3

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import pdb

from sim_modules.rnn import RNN

from sim_modules.testfuncs import gen_in_out_one_in_subs

data_path = "../../data/Gradient_Based_Adaptation/"

n_samples = 1

n_sweep = 1

gain_arr = np.linspace(0.9,0.9,n_sweep)

#tau_arr = np.array([1,5,10,15])
tau_arr = np.array([10,15])

T_sim = 1000000

T_test = 10000

for s in tqdm(range(n_samples)):

    for n in tqdm(range(tau_arr.shape[0])):

        for k in tqdm(range(n_sweep)):

            ### Global Gradient
            rnn = RNN(N=1000)
            '''
            # draw gains from rotationally invariant distribution
            rnn.a = np.random.normal(0.,1.,rnn.N)
            # "flip" negative values
            rnn.a  *= 2.*(rnn.a > 0.) - 1.
            # normalize to unit length times sqrt(N)
            rnn.a *= rnn.N**.5 / np.linalg.norm(rnn.a)
            '''
            #rescale
            rnn.a *= gain_arr[k]

            rnn.eps_a = 0.0002
            rnn.eps_w_out = 0.005

            u_in,u_out = gen_in_out_one_in_subs(T_sim,tau_arr[n])

            plt.ion()

            fig, ax = plt.subplots(3,1)

            plt.pause(0.1)

            #'''
            t,y,X,X_r,X_e,a,delta_a,w_out,E,W = rnn.learn_gain(u_in,u_out,mode='global_grad_local_gain',
                                                            mode_w_out="batch",
                                                               tau_batch_w_out=1.,
                                                               T_batch_w_out=rnn.N*5,
                                                               T_skip_rec=500,
                                                               fix_gain_radius=False,
                                                               show_progress=True,return_dyda=False,
                                                               randomize_RTRL_matrix=False,
                                                               ax_a=ax[0],ax_w_out=ax[1],ax_Err=ax[2])
            #'''
            '''
            rnn.a[:] = gain_arr[k]

            t,y,X,X_r,X_e,a,delta_a,w_out,E,W = rnn.learn_gain(u_in,u_out,mode='global_grad_global_gain',
                                                            mode_w_out="batch",
                                                               tau_batch_w_out=1.,T_skip_rec=500,
                                                               fix_gain_radius=False,
                                                               show_progress=True,return_dyda=False,
                                                               randomize_RTRL_matrix=True,
                                                               ax=ax[1])

            rnn.a[:] = gain_arr[k]

            t,y,X,X_r,X_e,a,delta_a,w_out,E,W = rnn.learn_gain(u_in,u_out,mode='local_grad_global_gain',
                                                            mode_w_out="batch",
                                                               tau_batch_w_out=1.,T_skip_rec=500,
                                                               fix_gain_radius=False,
                                                               show_progress=True,return_dyda=False,
                                                               randomize_RTRL_matrix=False,
                                                               ax=ax[2])
            '''
            pdb.set_trace()

pdb.set_trace()

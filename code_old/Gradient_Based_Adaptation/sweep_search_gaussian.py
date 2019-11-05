#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sim_modules.testfuncs import gen_in_out_one_in_subs
from sim_modules.rnn import RNN

n_sweep = 30

N_net = 500

sigm_e = np.linspace(0.,1.5,n_sweep+1)
sigm_y = np.linspace(0.,.9,n_sweep+1)

perf_xor = np.zeros((n_sweep,n_sweep))

specrad = np.ndarray((n_sweep,n_sweep))

tau_max = 15

t_prerun = 100

for k in tqdm(range(n_sweep)):
    for l in tqdm(range(n_sweep),disable=False):


        rnn = RNN(N=N_net,a_e = 0.9,a_r =0.9,y_std_target=sigm_y[l])

        y,X_r,X_e,a_r,a_e,b,y_mean,y_std = rnn.run_hom_adapt(sigm_e=sigm_e[k],T_skip_rec=10,show_progress=False)
        #y,X_r,X_e,a_r,a_e,b,y_mean,y_std = rnn.run_hom_adapt(u_in=u_in,T_skip_rec=10,show_progress=True)
        #'''
        for tau in range(tau_max):
            u_in, u_out = gen_in_out_one_in_subs(rnn.N*5+t_prerun,tau)
            u_in *= 2.*sigm_e[k]
            rnn.learn_w_out_trial(u_in,u_out,show_progress=True,T_prerun=t_prerun)

            u_in_test, u_out_test = gen_in_out_one_in_subs(rnn.N*3+t_prerun,tau)
            u_in_test *= 2.*sigm_e[k]
            u_pred = rnn.predict_data(u_in_test,show_progress=True)

            perf_xor[k,l] += np.corrcoef(u_out_test[t_prerun:],u_pred[t_prerun:])[1,0]**2.
        #'''
        specrad[k,l] = np.abs(np.linalg.eigvals((rnn.W.T * rnn.a_r).T)).max()

np.save("../../data/perf_xor_gaussian.npy",perf_xor)
np.save("../../data/specrad_gaussian.npy",specrad)

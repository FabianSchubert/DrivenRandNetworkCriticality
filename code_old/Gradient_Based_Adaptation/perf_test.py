#!/usr/bin/env python3

import numpy as np

from tqdm import tqdm

from sim_modules.rnn import RNN

from sim_modules.testfuncs import gen_in_out_one_in_subs

data_path = "../../data/Gradient_Based_Adaptation/"

N=50

n_samples = 10

n_sweep = 50

gain_arr = np.linspace(0.2,1.5,n_sweep)
tau_arr = np.array([1,5,10,15])
#tau_arr = np.array([15])

T_batch_w_out = N*5

T_sim = 50000

T_skip_rec = 1

T_rec = int(T_sim/T_skip_rec)

msqe = np.ndarray((n_samples,tau_arr.shape[0],n_sweep))


for s in tqdm(range(n_samples)):

    for n in tqdm(range(tau_arr.shape[0])):

        ### Local Gradient
        rnn = RNN(N=N)

        for k in tqdm(range(n_sweep)):

            rnn.a[:] = gain_arr[k]

            u_in_train, u_out_train = gen_in_out_one_in_subs(T_batch_w_out,tau_arr[n])

            rnn.learn_w_out(u_in_train,u_out_train,reg_fact=0.0001)

            u_in_test, u_out_test = gen_in_out_one_in_subs(T_sim,tau_arr[n])

            u_out_pred = rnn.predict_data(u_in_test)

            msqe[s,n,k] = ((u_out_test - u_out_pred)**2.).mean()


    np.save(data_path + "msqe_sweep_N_50_2.npy",msqe[:s+1,:,:])

import pdb
pdb.set_trace()

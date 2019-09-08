#!/usr/bin/env python3

import numpy as np

from echo_state_tests import gen_in_out_one_in_subs

from scipy.sparse import csr_matrix

from esn_module import esn

import pickle

import sys

from tqdm import tqdm

N = 1000

cf = .1
cf_in = 1.

sigm_w = 1.
sigm_w_in = 1.

W = np.random.normal(0.,sigm_w/(cf*N)**.5,(N,N))*(np.random.rand(N,N) <= cf)
W[range(0),range(0)] = 0.

W = csr_matrix(W)

WT = np.array(W.todense()).T
#WTSP = csr_matrix(WT)

w_in = np.random.normal(0.,sigm_w_in,(N))*(np.random.rand(N) <= cf_in)

###
eps_w_out = 0.01
eps_a = .5

## alpha = 0.: only local gradients, 1.: only pop. average gradient.
alpha = .9


###
T = 100000
T_skip_rec = 10
T_rec = int(T/T_skip_rec)
T_w_o_learn = 100
T_skip_w_o_learn = 100

t_ax = np.array(range(T_rec))*T_skip_rec

y_rec = np.ndarray((T_rec,N))
a_rec = np.ndarray((T_rec,N))
w_out_rec = np.ndarray((T_rec,N+1))

y_rec_w_out_learn = np.zeros((T_w_o_learn,N+1))

E_rec = np.ndarray((T_rec))
###
reg_fact = 0.01

### delay for xor-problem
tau = int(sys.argv[1])


u_in, u_out = gen_in_out_one_in_subs(T,5)
u_in = np.array([u_in]).T
u_out = np.array([u_out]).T

y = np.random.rand(N+1)-.5
y[0] = 1.

a = np.ones((N))*1.

w_out = np.random.rand(N+1)-.5
w_out[0] = 0.

H = np.array(W.todense())

G = np.zeros((N,N))

for t in tqdm(range(T)):

    X = W.dot(y[1:]) + w_in*u_in[t]

    y[1:] = np.tanh(a*X)

    y_rec_w_out_learn[:-1,:] = y_rec_w_out_learn[1:,:]
    y_rec_w_out_learn[-1,:] = y[:]

    O = y @ w_out

    if t>=T_w_o_learn:

        H = (WT * a * (1.-y[1:]**2.)).T

        G = H @ G

        G[range(N),range(N)] += (1.-y[1:]**2.)*X

        #nabl_w_out = (O - u_out[t])*y

        nabl_a = (O - u_out[t])*(w_out[1:] @ G)
        #nabl_a = np.asarray((O - u_out[t])*(w_out[1:] @ G))[0,:]

        #w_out -= eps_w_out * nabl_w_out

        #import pdb
        #pdb.set_trace()

        a -= eps_a * (nabl_a * (1.-alpha) + alpha * nabl_a.mean())

        a = np.maximum(a,0.001)

    if t%T_skip_w_o_learn == 0 and t>=T_w_o_learn:

        w_out = (np.linalg.inv(y_rec_w_out_learn.T @ y_rec_w_out_learn + reg_fact*np.eye(N+1)) @ y_rec_w_out_learn.T @ u_out[t-T_w_o_learn+1:t+1,:])[:,0]


    ####
    if t%T_skip_rec == 0:

        t_rec = int(t/T_skip_rec)

        y_rec[t_rec,:] = y[1:]
        a_rec[t_rec,:] = a
        w_out_rec[t_rec,:] = w_out

        E_rec[t_rec] = (O - u_out[t])**2./2.

w_out = (np.linalg.inv(y_rec_w_out_learn.T @ y_rec_w_out_learn + reg_fact*np.eye(N+1)) @ y_rec_w_out_learn.T @ u_out[t-T_w_o_learn+1:t+1,:])[:,0]


ESN = esn(N=N)
ESN.W = csr_matrix(W)
ESN.w_in = np.array([w_in]).T
ESN.w_out = np.array([w_out])
ESN.gain = a

save_dict = {"ESN":ESN,"tau":tau}

pickle.dump(save_dict, open( "../data/RTRL/save_dict_tau"+"_"+ str(tau) + ".p", "wb" ))

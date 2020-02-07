import numpy as np
from tqdm import tqdm

from stdParams import *
import os

from datetime import datetime

import sys

N = 500

n_samples = 1

cf_w = .1
cf_w_in = 1.

sigm_w_e = .5

eps_a = 1e-3
eps_b = 10e-4

eps_mu = 10e-4
eps_var = 10e-3

mu_y_target = np.ones((N))*0.

a_init = 1.5
X_r_norm_init_span = 100.

#r_target = .9

T = int(3e4)
T_skip_rec = 1
T_rec = int(T/T_skip_rec)

t_arr = np.arange(T_rec)*T_skip_rec


### recording
a_rec = np.ones((n_samples,T_rec,N))
b_rec = np.ones((T_rec,N))

y_rec = np.ones((T_rec,N))
y_norm_rec = np.ones((n_samples,T_rec))
X_r_rec = np.ones((T_rec,N))
X_e_rec = np.ones((T_rec,N))

mu_y_rec = np.ones((T_rec,N))
mu_X_e_rec = np.ones((T_rec,N))
Var_y_rec = np.ones((T_rec,N))
Var_X_e_rec = np.ones((T_rec,N))
###

for k in tqdm(range(n_samples)):

    W = np.random.normal(0.,1./(cf_w*N)**.5,(N,N)) * (np.random.rand(N,N) <= cf_w)
    W[range(N),range(N)] = 0.

    W = W/np.max(np.abs(np.linalg.eigvals(W)))

    W_av = 1.*(W!=0.)
    W_av = (W_av.T / W_av.sum(axis=1)).T

    a = np.ones((N))*a_init
    b = np.zeros((N))

    if sigm_w_e > 0.:
        w_in = np.random.normal(0.,sigm_w_e,(N)) * (np.random.rand(N) <= cf_w_in)
        #w_in = np.ones((N,1))*sigm_w_e
    else:
        w_in = np.zeros((N,1))

    #u_in = (np.random.rand(1,T) >= .5)*2.-1.

    y = np.ndarray((N))
    X_r = np.ndarray((N))
    X_e = np.ndarray((N))
    #X_e = (w_in @ u_in).T
    #X_e = np.random.normal(0.,1.,(T,N)) * w_in[:,0]
    #X_e = np.random.normal(0.,.25,(T,N))

    mu_y = np.ndarray((N))
    mu_X_e = np.ndarray((N))
    Var_y = np.ndarray((N))
    Var_X_e = np.ndarray((N))



    ### first time step
    X_e[:] = w_in * np.random.normal(0.,1.,(N))
    X_r[:] = (np.random.rand(N)-.5)
    X_r[:] *= np.random.rand()*X_r_norm_init_span/np.linalg.norm(X_r)
    y[:] = np.tanh(X_r[:] + X_e[:])


    mu_y[:] = y
    mu_X_e[:] = X_e
    Var_y[:] = .25
    Var_X_e[:] = .25

    #### Recording
    a_rec[k,0,:] = a
    b_rec[0,:] = b

    y_rec[0,:] = y
    y_norm_rec[k,0] = np.linalg.norm(y)
    X_r_rec[0,:] = X_r
    X_e_rec[0,:] = X_e

    mu_y_rec[0,:] = mu_y
    mu_X_e_rec[0,:] = mu_X_e
    Var_y_rec[0,:] = Var_y
    Var_X_e_rec[0,:] = Var_X_e
    ####
    ###

    for t in range(1,T):

        y_prev = y[:]

        X_r[:] = a[:] * (W @ y[:])
        X_e[:] = w_in * np.random.normal(0.,1.,(N))

        y[:] = np.tanh(X_r + X_e - b)

        mu_y[:] = (1.-eps_mu)*mu_y + eps_mu * y
        mu_X_e[:] = (1.-eps_mu)*mu_X_e + eps_mu * X_e

        Var_y[:] = (1.-eps_var)*Var_y + eps_var * (y - mu_y)**2.
        Var_X_e[:] = (1.-eps_var)*Var_X_e + eps_var * (X_e - mu_X_e)**2.

        #y_squ_targ = 1.-1./(1.+2.*Var_y.mean() + 2.*Var_X_e)**.5

        #a = a + eps_a * a * ((y**2.).mean() - (X_r**2.).mean())
        a = a + eps_a * a * (y_prev**2. - X_r**2.)
        #a = a + eps_a * (W_av @ (y_prev**2.) - X_r**2.)
        #a = a + eps_a * ((y**2.) - (X_r**2.))
        b = b + eps_b * (y - mu_y_target)

        a = np.maximum(0.001,a)

        if t%T_skip_rec == 0:
            t_rec = int(t/T_skip_rec)

            #### Recording
            a_rec[k,t_rec,:] = a
            b_rec[t_rec,:] = b

            y_rec[t_rec,:] = y
            X_r_rec[t_rec,:] = X_r
            X_e_rec[t_rec,:] = X_e

            mu_y_rec[t_rec,:] = mu_y
            mu_X_e_rec[t_rec,:] = mu_X_e
            Var_y_rec[t_rec,:] = Var_y
            Var_X_e_rec[t_rec,:] = Var_X_e
            ####
    y_norm_rec[k,:] = np.linalg.norm(y_rec,axis=1)

if not(os.path.isdir(os.path.join(DATA_DIR,'heterogeneous_independent_gaussian_input_ESN/alt_hom_regulation/'))):
    os.makedirs(os.path.join(DATA_DIR,'heterogeneous_independent_gaussian_input_ESN/alt_hom_regulation/'))

np.savez(os.path.join(DATA_DIR,'heterogeneous_independent_gaussian_input_ESN/alt_hom_regulation/alt_hom_regulation_'+str(datetime.now().isoformat())+'.npz'),
        a=a_rec,
        b=b_rec,
        W=W,
        y_norm=y_norm_rec,
        y=y_rec,
        N=N,
        n_samples=n_samples,
        cf_w = cf_w,
        cf_w_in = cf_w_in,
        sigm_w_e =sigm_w_e,
        eps_a = eps_a,
        eps_b = eps_b,
        mu_y_target = mu_y_target,
        X_r=X_r_rec)

import numpy as np

from src.rnn import RNN

from src.testfuncs import gen_in_out_one_in_subs

from tqdm import tqdm

from stdParams import *
import os

from datetime import datetime

import sys


try:
    N = int(sys.argv[1])
except:
    N = 500
    #N = 1000

try:
    n_sweep_sigm_e = int(sys.argv[2])
    sigm_e = .5*np.array(range(n_sweep_sigm_e))
except:
    n_sweep_sigm_e = 5
    sigm_e = np.array([0.,0.05,.5,1.,1.5])

try:
    n_sweep_sigm_t = int(sys.argv[3])
except:
    n_sweep_sigm_t = 30

sigm_t = np.linspace(0.,0.9,n_sweep_sigm_t)



T_run_adapt = 200000
T_run_sample = 1000
T_prerun_sample = 100

y = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,T_run_sample,N))
X_r = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,T_run_sample,N))
X_e = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,T_run_sample,N))

W = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,N,N))
a = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,N))
b = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,N))

for k in tqdm(range(n_sweep_sigm_e)):
    for l in tqdm(range(n_sweep_sigm_t)):

        rnn = RNN(N=N,y_mean_target=0.,y_std_target=sigm_t[l])

        rnn.w_in = np.ones((rnn.N,1))

        u_in_adapt,u_out = gen_in_out_one_in_subs(T_run_adapt,1)
        u_in_adapt *= 2.*sigm_e[k]

        adapt = rnn.run_hom_adapt(u_in=u_in_adapt,T_skip_rec=1000)

        u_in_sample,u_out = gen_in_out_one_in_subs(T_run_sample+T_prerun_sample,1)
        u_in_sample *= 2.*sigm_e[k]

        y_res,X_r_res,X_e_res = rnn.run_sample(u_in=u_in_sample)

        y[k,l,:,:] = y_res[T_prerun_sample:,:]
        X_r[k,l,:,:] = X_r_res[T_prerun_sample:,:]
        X_e[k,l,:,:] = X_e_res[T_prerun_sample:,:]

        W[k,l,:,:] = rnn.W
        a[k,l,:] = rnn.a_r
        b[k,l,:] = rnn.b

if not(os.path.isdir(os.path.join(DATA_DIR,'homogeneous_identical_binary_input_ESN/N_' + str(N)))):
    os.makedirs(os.path.join(DATA_DIR,'homogeneous_identical_binary_input_ESN/N_' + str(N)))

np.savez(os.path.join(DATA_DIR,'homogeneous_identical_binary_input_ESN/N_' + str(N) + '/param_sweep_'+str(datetime.now().isoformat())+'.npz'),
        sigm_t=sigm_t,
        sigm_e=sigm_e,
        y=y,
        X_r=X_r,
        X_e=X_e,
        W=W,
        a=a,
        b=b)

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
except:
    n_sweep_sigm_e = 30

sigm_e = np.linspace(0.,1.5,n_sweep_sigm_e)

try:
    n_sweep_sigm_t = int(sys.argv[3])
except:
    n_sweep_sigm_t = 30

sigm_t = np.linspace(0.,0.9,n_sweep_sigm_t)

T_run_adapt = 200000
T_prerun = 100
T_run_learn = 10*N
T_run_test = 10*N

tau_max = 15

y_mean_target = 0.05

MC = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,tau_max))

W = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,N,N))
a = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,N))
b = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,N))

for k in tqdm(range(n_sweep_sigm_e)):
    for l in tqdm(range(n_sweep_sigm_t)):

        rnn = RNN(N=N,y_mean_target=y_mean_target,y_std_target=sigm_t[l])

        u_in_adapt,u_out = gen_in_out_one_in_subs(T_run_adapt,1)
        u_in_adapt *= 2.*sigm_e[k]

        adapt = rnn.run_hom_adapt(u_in=u_in_adapt,T_skip_rec=1000)

        W[k,l,:,:] = rnn.W
        a[k,l,:] = rnn.a_r
        b[k,l,:] = rnn.b

        for tau in range(tau_max):

            u_in_learn,u_out_learn = gen_in_out_one_in_subs(T_run_learn+T_prerun,tau)
            u_in_learn *= 2.*sigm_e[k]

            rnn.learn_w_out_trial(u_in_learn,u_out_learn,reg_fact=.01,show_progress=False,T_prerun=T_prerun)

            u_in_test,u_out_test = gen_in_out_one_in_subs(T_run_test+T_prerun,tau)
            u_in_test *= 2.*sigm_e[k]

            u_out_pred = rnn.predict_data(u_in_test)

            MC[k,l,tau] = np.corrcoef(u_out_test[T_prerun:],u_out_pred[T_prerun:])[0,1]**2.

if not(os.path.isdir(os.path.join(DATA_DIR,'heterogeneous_identical_binary_input_ESN/'))):
    os.makedirs(os.path.join(DATA_DIR,'heterogeneous_identical_binary_input_ESN/'))

np.savez(os.path.join(DATA_DIR,'heterogeneous_identical_binary_input_ESN/param_sweep_performance_'+str(datetime.now().isoformat())+'.npz'),
        sigm_t=sigm_t,
        sigm_e=sigm_e,
        W=W,
        a=a,
        b=b,
        MC=MC)

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
T_run_ESP = 10*N
T_sample_variance = 10*N

d_init_ESP = 10.**-3.


tau_max = 15

y_mean_target = 0.05

MC = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,tau_max))

ESP = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,T_run_ESP))

W = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,N,N))
a = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,N))
b = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,N))

sigm_x_r_adapt = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,N))
sigm_x_r_test = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,N))

sigm_x_e_adapt = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,N))
sigm_x_e_test = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,N))

for k in tqdm(range(n_sweep_sigm_e)):
    for l in tqdm(range(n_sweep_sigm_t)):

        rnn = RNN(N=N,y_mean_target=y_mean_target,y_std_target=sigm_t[l])

        sigm_e_dist = np.abs(np.random.normal(0.,sigm_e[k],(rnn.N)))

        adapt = rnn.run_hom_adapt(u_in=None,sigm_e=sigm_e_dist,T=T_run_adapt,T_skip_rec=1000)

        #run sample after adaptation, USING THE INPUT STATISTICS OF THE ADAPTATION!!
        y, X_r, X_e = rnn.run_sample(u_in=None,sigm_e=sigm_e_dist,T=T_sample_variance,show_progress=True)

        W[k,l,:,:] = rnn.W
        a[k,l,:] = rnn.a_r
        b[k,l,:] = rnn.b

        for tau in range(tau_max):

            u_in_learn,u_out_learn = gen_in_out_one_in_subs(T_run_learn+T_prerun,tau)
            u_in_learn *= sigm_e[k]

            rnn.learn_w_out_trial(u_in_learn,u_out_learn,reg_fact=.01,show_progress=False,T_prerun=T_prerun)

            u_in_test,u_out_test = gen_in_out_one_in_subs(T_run_test+T_prerun,tau)
            u_in_test *= sigm_e[k]

            u_out_pred = rnn.predict_data(u_in_test)

            MC[k,l,tau] = np.corrcoef(u_out_test[T_prerun:],u_out_pred[T_prerun:])[0,1]**2.

        ### test ESP
        u_in_ESP,u_out_ESP = gen_in_out_one_in_subs(T_run_ESP,0)
        u_in_ESP *= sigm_e[k]

        X_r_init = np.random.normal(0.,1.,(N))
        delta_X_r_init = np.random.normal(0.,1.,(N))
        delta_X_r_init = d_init_ESP / np.linalg.norm(delta_X_r_init)

        y_ESP_1, X_r_ESP_1, X_e_ESP_1 = rnn.run_sample(u_in=u_in_ESP,X_r_init=X_r_init,show_progress=True)
        y_ESP_2, X_r_ESP_2, X_e_ESP_2 = rnn.run_sample(u_in=u_in_ESP,X_r_init=X_r_init+delta_X_r_init,show_progress=True)

        d = np.linalg.norm(y_ESP_1-y_ESP_2,axis=1)

        ESP[k,l,:] = d

        #run test sample with xor_input
        u_in_test,u_out_test = gen_in_out_one_in_subs(T_sample_variance,0)
        u_in_test *= sigm_e[k]
        y, X_r, X_e = rnn.run_sample(u_in=u_in_test,show_progress=True)

        sigm_x_r_test[k,l,:] = X_r.std(axis=0)
        sigm_x_e_test[k,l,:] = X_e.std(axis=0)

if not(os.path.isdir(os.path.join(DATA_DIR,'heterogeneous_independent_gaussian_input_ESN/'))):
    os.makedirs(os.path.join(DATA_DIR,'heterogeneous_independent_gaussian_input_ESN/'))

np.savez(os.path.join(DATA_DIR,'heterogeneous_independent_gaussian_input_ESN/param_sweep_performance_'+str(datetime.now().isoformat())+'.npz'),
        sigm_t=sigm_t,
        sigm_e=sigm_e,
        sigm_x_r_adapt=sigm_x_r_adapt,
        sigm_x_r_test=sigm_x_r_test,
        sigm_x_e_adapt=sigm_x_e_adapt,
        sigm_x_e_test=sigm_x_e_test,
        W=W,
        a=a,
        b=b,
        MC=MC,
        ESP=ESP)
